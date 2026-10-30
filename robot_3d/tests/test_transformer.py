"""Train/reload a tiny local encoder and check complete 3-D warm-start inputs."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from q_transformer.data import generate_q_data
from robot_3d.generate_robot_3d_q_data import training_maps
from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP
from robot_3d.train_robot_3d_transformer import main as train_transformer
from Tabular_Agent import Tabular_Agent


HAS_TRANSFORMER = all(importlib.util.find_spec(name)
                      for name in ("torch", "transformers", "accelerate"))


@unittest.skipUnless(HAS_TRANSFORMER, "Install requirement.txt for model checks")
class TransformerTests(unittest.TestCase):
    def test_training_reload_real_warm_start_full_shape_and_truncation_guard(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast

        from q_transformer.model import QValueTransformer
        from robot_3d.transformer import QTransformerPredictor

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            encoder_path = directory / "encoder"
            encoder_path.mkdir()
            words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            words += list(map(str, range(10)))
            words += ["free", "goal", "obstacle", "cells", "start", "grid", "none", "r", "c",
                      "x", "y", "z", "#", "=", ".", ",", "(", ")", ":", "-", "infinity"]
            (encoder_path / "vocab.txt").write_text("\n".join(words), encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(encoder_path / "vocab.txt"))
            tokenizer.save_pretrained(encoder_path)
            config = BertConfig(
                vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=256,
                hidden_dropout_prob=0, attention_probs_dropout_prob=0,
            )
            BertModel(config).save_pretrained(encoder_path)
            template = Robot3DMotionMDP(
                x_max=2, y_max=2, z_max=2, grid_shape=(2, 2, 2),
                start_cell=(0, 0, 0), goal_cell=(1, 1, 1), gamma=0.9,
                movement_reward=0, goal_reward=1,
            )
            maps = list(training_maps(template, 4, n_obstacles=(0, 0), start_goal_seperation=1))
            output = directory / "saved_results/q_transformer"
            output.mkdir(parents=True)
            dataset = output / "q_data.npz"
            generate_q_data(maps, (0, 0.3), (0, 0.6), dataset)
            script = str(directory / "train.py")
            with (
                patch("robot_3d.train_robot_3d_transformer.__file__", script),
                patch("q_transformer.train.torch.cuda.is_available", return_value=False),
            ):
                history = train_transformer(
                    epochs=2, batch_size=4, learning_rate=0.01,
                    model_name=str(encoder_path), max_length=256,
                )
            self.assertTrue(np.isfinite(history[-1]["train_loss"]))
            predictor = QTransformerPredictor(output / "checkpoint", device="cpu")
            initial = predictor(template, 0.3, 0.6)
            self.assertEqual(initial.shape, (8, 6))
            np.testing.assert_array_equal(initial[template.terminal_mask], 0)
            agent = Tabular_Agent(template, lr_init=0.5, R=0.3, C=0.6, seed=7)
            agent.robust_q_learning(max_training_episodes=5000, initial_Q=initial)
            self.assertFalse(agent.training_diagnostics["zero_initialized"])
            self.assertTrue(agent.training_diagnostics["converged"])
            self.assertEqual(agent.evaluation_return[0], initial[0].max())
            with patch.object(template, "q_transformer_context", return_value="free " * 600):
                with self.assertRaisesRegex(ValueError, "complete 3-D prompt needs"):
                    predictor(template, 0.3, 0.6)

            full_mdp = Robot3DMotionMDP(
                x_max=10, y_max=10, z_max=10, grid_shape=(10, 10, 10),
                start_cell=(0, 0, 0), goal_cell=(9, 9, 9), gamma=0.9,
                movement_reward=0, goal_reward=1,
                obstacles=[(x, y, 5) for y in range(3) for x in range(10)],
            )
            full_checkpoint = directory / "full_shape"
            QValueTransformer(BertModel(config), 1000, 6).save_checkpoint(
                full_checkpoint, tokenizer, 256, {},
            )
            full_predictor = QTransformerPredictor(full_checkpoint, device="cpu")
            full_Q = full_predictor(full_mdp, 0.12345678901234566, 1.2345678901234567)
            self.assertEqual(full_Q.shape, (1000, 6))
            self.assertTrue(np.all(np.isfinite(full_Q)))
            np.testing.assert_array_equal(full_Q[full_mdp.terminal_mask], 0)

            np.savez_compressed(
                dataset, descriptions=np.array(["free " * 600]), map_index=np.array([0, 0]),
                R=np.array([0, 0.3]), C=np.array([0, 0.6]), Q=np.zeros((2, 8, 6)),
            )
            with (
                patch("robot_3d.train_robot_3d_transformer.__file__", script),
                patch("q_transformer.train.train_q_transformer", side_effect=AssertionError),
            ):
                with self.assertRaisesRegex(ValueError, "complete 3-D prompt needs"):
                    train_transformer(model_name=str(encoder_path), max_length=256)


if __name__ == "__main__":
    unittest.main()
