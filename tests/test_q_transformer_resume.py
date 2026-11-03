"""Exercise checkpoint continuation and interruption using a tiny local BERT model."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np


HAS_TRANSFORMER = all(
    importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate")
)


@unittest.skipUnless(HAS_TRANSFORMER, "Install requirement.txt for model checks")
class TransformerResumeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.encoder = self.root / "encoder"
        self.encoder.mkdir()
        words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "free", "grid"]
        (self.encoder / "vocab.txt").write_text("\n".join(words), encoding="utf-8")
        BertTokenizerFast(vocab_file=str(self.encoder / "vocab.txt")).save_pretrained(self.encoder)
        config = BertConfig(
            vocab_size=len(words), hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, max_position_embeddings=64,
        )
        BertModel(config).save_pretrained(self.encoder)
        self.dataset = self.root / "data.npz"
        np.savez_compressed(
            self.dataset, descriptions=np.array(["free grid"] * 4),
            map_index=np.repeat(np.arange(4), 4), R=np.tile([0, 0.2, 0.3, 0.5], 4),
            C=np.ones(16), Q=np.random.default_rng(1).uniform(size=(16, 2, 6)),
        )

    def train(self, output, **kwargs):
        from transformers import EarlyStoppingCallback

        from q_transformer.train import train_q_transformer

        options = dict(
            eval_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="rmse",
            greater_is_better=False, save_total_limit=2, warmup_ratio=0.25, disable_tqdm=True,
        )
        options.update(kwargs.pop("training_options", {}))
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        callbacks.extend(kwargs.pop("callbacks", []))
        rates = dict(learning_rate=0.001, head_learning_rate=0.003, position_learning_rate=0.002)
        rates.update({name: kwargs.pop(name) for name in rates if name in kwargs})
        return train_q_transformer(
            self.dataset, output, model_name=str(self.encoder), batch_size=2, max_length=64,
            split_by_map=True, device="cpu", training_options=options, callbacks=callbacks,
            **rates, **kwargs,
        )

    def test_restart_preserves_adam_and_best_model_and_can_resume_again(self):
        import torch
        from transformers import TrainerCallback

        from q_transformer.model import QTransformerPredictor

        class Probe(TrainerCallback):
            def __init__(self, stop_at=None):
                self.stop_at = stop_at
                self.steps = []
                self.rates = []

            def on_train_begin(
                self, args, state, control, model, optimizer, lr_scheduler, **kwargs,
            ):
                self.start_step = state.global_step
                self.weights = {key: value.clone() for key, value in model.state_dict().items()}
                self.optimizer = copy.deepcopy(optimizer.state_dict())
                self.schedule = copy.deepcopy(lr_scheduler.state_dict())
                self.best_checkpoint = state.best_model_checkpoint

            def on_step_end(self, args, state, control, lr_scheduler, **kwargs):
                self.steps.append(state.global_step)
                self.rates.append(lr_scheduler.get_last_lr())
                if state.global_step == self.stop_at:
                    control.should_training_stop = True

            def on_evaluate(self, args, state, control, metrics, **kwargs):
                metrics["eval_rmse"] = state.best_metric + 1.0

        original = self.root / "original"
        self.train(
            original, epochs=2, learning_rate=0.002,
            head_learning_rate=0.006, position_learning_rate=0.004,
        )
        checkpoint = original / "checkpoint-12"
        state_path = checkpoint / "trainer_state.json"
        state = json.loads(state_path.read_text())
        best_name = Path(state["best_model_checkpoint"]).name
        baseline = state["best_metric"]
        state["best_model_checkpoint"] = f"/old/server/path/{best_name}"
        state["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"].update(
            early_stopping_patience_counter=4,
        )
        state_path.write_text(json.dumps(state), encoding="utf-8")
        saved_model = torch.load(checkpoint / "model.pt", weights_only=True)
        saved_optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
        saved_scheduler = torch.load(checkpoint / "scheduler.pt", weights_only=True)
        self.assertTrue(all(rate == 0 for rate in saved_scheduler["_last_lr"]))
        reference, interrupted = self.root / "reference", self.root / "interrupted"
        shutil.copytree(original, reference)
        shutil.copytree(original, interrupted)

        observer = Probe()
        self.train(
            reference, epochs=5, resume_from_checkpoint=reference / "checkpoint-12",
            restart_lr_schedule=True, callbacks=[observer],
        )
        self.assertEqual(observer.start_step, 12)
        self.assertEqual(observer.steps, list(range(13, 31)))
        for key, value in observer.weights.items():
            torch.testing.assert_close(value, saved_model["state_dict"][key], rtol=0, atol=0)
        for key, values in saved_optimizer["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(observer.optimizer["state"][key][name], value)
        self.assertEqual(observer.schedule["start_step"], 12)
        self.assertEqual(observer.schedule["num_training_steps"], 18)
        self.assertEqual(observer.schedule["num_warmup_steps"], 5)
        self.assertEqual(observer.schedule["last_epoch"], 0)
        expected_peaks = [0.001, 0.002, 0.001, 0.003, 0.003]
        self.assertEqual(observer.schedule["base_lrs"], expected_peaks)
        np.testing.assert_allclose(expected_peaks, np.array(saved_scheduler["base_lrs"]) / 2)
        np.testing.assert_allclose(np.max(observer.rates, axis=0), expected_peaks)
        self.assertTrue(all(rate == 0 for rate in observer.rates[-1]))
        self.assertEqual(Path(observer.best_checkpoint), reference / best_name)
        final_state = json.loads((reference / "checkpoint-30/trainer_state.json").read_text())
        self.assertEqual(final_state["best_metric"], baseline)
        self.assertEqual(Path(final_state["best_model_checkpoint"]), reference / best_name)
        self.assertEqual(len(list(reference.glob("checkpoint-*"))), 2)
        final_model = torch.load(reference / "model.pt", weights_only=True)
        best_model = torch.load(reference / best_name / "model.pt", weights_only=True)
        for key, value in final_model["state_dict"].items():
            torch.testing.assert_close(value, best_model["state_dict"][key], rtol=0, atol=0)
        predictor = QTransformerPredictor(reference, device="cpu")
        self.assertEqual((predictor.model.n_states, predictor.model.n_actions), (2, 6))

        self.train(
            interrupted, epochs=5, resume_from_checkpoint=interrupted / "checkpoint-12",
            restart_lr_schedule=True, callbacks=[Probe(stop_at=24)],
        )
        continued = Probe()
        self.train(
            interrupted, epochs=5, resume_from_checkpoint=interrupted / "checkpoint-24",
            callbacks=[continued],
        )
        self.assertEqual(continued.start_step, 24)
        self.assertEqual(continued.steps, list(range(25, 31)))
        self.assertEqual(continued.schedule["start_step"], 12)
        self.assertEqual(continued.schedule["last_epoch"], 12)
        self.assertGreater(max(continued.schedule["_last_lr"]), 0)
        expected = torch.load(reference / "checkpoint-30/model.pt", weights_only=True)
        actual = torch.load(interrupted / "checkpoint-30/model.pt", weights_only=True)
        for key, value in expected["state_dict"].items():
            torch.testing.assert_close(value, actual["state_dict"][key], rtol=0, atol=0)

    def test_interrupted_original_schedule_matches_uninterrupted_training(self):
        import torch
        from transformers import TrainerCallback

        class Stop(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 12:
                    control.should_training_stop = True

        reference, interrupted = self.root / "reference", self.root / "interrupted"
        self.train(reference, epochs=3)
        self.train(interrupted, epochs=3, callbacks=[Stop()])
        history = self.train(
            interrupted, epochs=3, resume_from_checkpoint=interrupted / "checkpoint-12",
        )
        self.assertEqual(history[-1]["epoch"], 3)
        expected = torch.load(reference / "checkpoint-18/model.pt", weights_only=True)
        actual = torch.load(interrupted / "checkpoint-18/model.pt", weights_only=True)
        for key, value in expected["state_dict"].items():
            torch.testing.assert_close(value, actual["state_dict"][key], rtol=0, atol=0)
        schedule = torch.load(interrupted / "checkpoint-18/scheduler.pt", weights_only=True)
        self.assertEqual(schedule["start_step"], 0)
        self.assertEqual(schedule["last_epoch"], 18)
        self.assertTrue(all(rate == 0 for rate in schedule["_last_lr"]))

    def test_invalid_resume_settings_fail_clearly(self):
        from q_transformer.train import train_q_transformer

        with self.assertRaisesRegex(ValueError, "requires a checkpoint"):
            self.train(self.root / "missing", epochs=2, restart_lr_schedule=True)
        with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoint file"):
            self.train(
                self.root / "missing", epochs=2, resume_from_checkpoint=self.root / "missing",
            )
        output = self.root / "checkpoint"
        self.train(output, epochs=1)
        checkpoint = output / "checkpoint-6"
        with self.assertRaisesRegex(ValueError, "extending the training budget"):
            self.train(output, epochs=2, resume_from_checkpoint=checkpoint)
        with self.assertRaisesRegex(ValueError, "must exceed the saved global step"):
            self.train(
                output, epochs=1, resume_from_checkpoint=checkpoint, restart_lr_schedule=True,
            )
        with self.assertRaisesRegex(ValueError, "original seed"):
            self.train(
                output, epochs=2, seed=1,
                resume_from_checkpoint=checkpoint, restart_lr_schedule=True,
            )
        with self.assertRaisesRegex(ValueError, "token limit"):
            train_q_transformer(
                self.dataset, output, epochs=2, batch_size=2, max_length=32, device="cpu",
                resume_from_checkpoint=checkpoint, restart_lr_schedule=True,
            )
        with self.assertRaisesRegex(ValueError, "number of training processes"):
            (checkpoint / "rng_state.pth").rename(checkpoint / "rng_state_0.pth")
            self.train(
                output, epochs=2, resume_from_checkpoint=checkpoint, restart_lr_schedule=True,
            )


if __name__ == "__main__":
    unittest.main()
