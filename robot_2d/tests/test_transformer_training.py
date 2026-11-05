"""Check 2-D validation on held-out maps and export of the best trained checkpoint."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from q_transformer.data import generate_q_data
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from tests.script_runner import run_script


HAS_TRANSFORMER = all(importlib.util.find_spec(name)
                      for name in ("torch", "transformers", "accelerate"))


@unittest.skipUnless(HAS_TRANSFORMER, "Install requirement.txt for model checks")
class TransformerTrainingTests(unittest.TestCase):
    def test_validation_holds_out_maps_and_exports_the_best_checkpoint(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast, EarlyStoppingCallback

        from q_transformer.model import QTransformerPredictor
        from q_transformer.train import _QTrainer

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            encoder_path = directory / "encoder"
            encoder_path.mkdir()
            words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            words += list(map(str, range(10))) + ["free", "goal", "grid", "r", "c", ".", ","]
            (encoder_path / "vocab.txt").write_text("\n".join(words), encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(encoder_path / "vocab.txt"))
            tokenizer.save_pretrained(encoder_path)
            config = BertConfig(
                vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=512,
                hidden_dropout_prob=0, attention_probs_dropout_prob=0,
            )
            BertModel(config).save_pretrained(encoder_path)
            template = Robot2DMotionMDP(
                x_max=4, y_max=3, grid_shape=(4, 3), start_cell=(0, 0), goal_cell=(3, 2),
                gamma=0.9, movement_reward=0.0, goal_reward=1.0,
            )
            maps = []
            for goal in template.states[1:11]:
                mdp = template.clone(nominal=True)
                mdp.goal_state = tuple(map(int, goal))
                mdp.set_obstacles([])
                mdp.reset()
                maps.append(mdp)
            dataset_path = directory / "data.npz"
            generate_q_data(maps, (0.0, 0.2), (0.0, 0.75), dataset_path)
            trainers = []

            def capture_trainer(*args, **kwargs):
                trainer = _QTrainer(*args, **kwargs)
                trainers.append(trainer)
                return trainer

            checkpoint = directory / "checkpoint"
            with patch("q_transformer.train._QTrainer", side_effect=capture_trainer):
                namespace = run_script(
                    "robot_2d.train_robot_2d_transformer",
                    dataset_path=dataset_path, checkpoint_path=checkpoint,
                    model_name=str(encoder_path), epochs=2, learning_rate=0.01, device="cpu",
                )
            trainer = trainers[0]
            with np.load(dataset_path, allow_pickle=False) as data:
                train_maps = set(data["map_index"][trainer.train_dataset.indices])
                validation_maps = set(data["map_index"][trainer.eval_dataset.indices])
            self.assertEqual((len(train_maps), len(validation_maps)), (9, 1))
            self.assertFalse(train_maps & validation_maps)
            self.assertEqual((len(trainer.train_dataset), len(trainer.eval_dataset)), (36, 4))
            evaluations = [row for row in namespace["history"] if "eval_rmse" in row]
            self.assertEqual([row["epoch"] for row in evaluations], [1.0, 2.0])
            self.assertTrue(all(np.isfinite(row["eval_rmse"]) for row in evaluations))
            best = min(row["eval_rmse"] for row in evaluations)
            self.assertAlmostEqual(trainer.state.best_metric, best)
            exported = torch.load(checkpoint / "model.pt", weights_only=True)
            selected = torch.load(
                Path(trainer.state.best_model_checkpoint) / "model.pt", weights_only=True,
            )
            for name, weights in exported["state_dict"].items():
                torch.testing.assert_close(weights, selected["state_dict"][name], rtol=0, atol=0)
            self.assertAlmostEqual(exported["training"]["best_metric"], best)
            self.assertEqual(exported["max_length"], 512)
            self.assertEqual(trainer.model.encoder.config.max_position_embeddings, 512)
            callbacks = [callback for callback in trainer.callback_handler.callbacks
                         if isinstance(callback, EarlyStoppingCallback)]
            self.assertEqual(len(callbacks), 1)
            self.assertEqual(callbacks[0].early_stopping_patience, 5)
            self.assertEqual(callbacks[0].early_stopping_threshold, 1e-5)
            self.assertTrue(trainer.args.use_cpu)
            self.assertFalse(trainer.args.fp16 or trainer.args.bf16)
            self.assertIsNone(trainer.args.tf32)
            predicted = QTransformerPredictor(checkpoint, device="cpu")(maps[0], 0.2, 0.75)
            self.assertEqual(predicted.shape, (12, 4))
            self.assertTrue(np.all(np.isfinite(predicted)))
            np.testing.assert_array_equal(predicted[maps[0].terminal_mask], 0.0)


if __name__ == "__main__":
    unittest.main()
