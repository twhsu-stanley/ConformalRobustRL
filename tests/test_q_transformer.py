"""Shared data, training, and inference checks using a small three-dimensional MDP."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np

from q_transformer.data import format_prompt, generate_q_data, split_samples
from Tabular_Agent import Tabular_Agent

HAS_TRANSFORMER = all(
    importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate")
)


class ThreeDimensionalMDP(gym.Env):
    """Two cells and six actions suffice to check that no 2-D shape is hard-coded."""

    def __init__(self, terminal_row=1):
        self.states = np.array([(0, 0, 0), (1, 0, 0)])
        self.action_space = gym.spaces.Discrete(6)
        self.terminal_mask = np.arange(2) == terminal_row
        self.trainable_states = self.states[~self.terminal_mask]
        self.start_state = self.trainable_states[0]
        self.goal_state = self.states[terminal_row]

    def state_index(self, state):
        return int(state[0])

    def values_at(self, state, values):
        row = self.state_index(state)
        return 0.0 if self.terminal_mask[row] else values[row]

    def worst_case_value(self, state, C, values):
        return min(0.0, values[~self.terminal_mask].min())

    def nominal_next_state(self, state, action):
        return self.goal_state

    def reward(self, state, action):
        return float(not self.terminal_mask[self.state_index(state)])

    def q_transformer_context(self):
        start, goal = tuple(map(int, self.start_state)), tuple(map(int, self.goal_state))
        return (
            "You are given a grid with size 2x1x1.\n"
            "The maze is represented as layers along the z-axis.\n"
            "0 = free space, 1 = obstacle.\n\n"
            "Layer z = 0\n0 0\n"
            f"Start at {start} and move to {goal}.\n"
        )

    def reset(self, *, seed=None, options=None):
        self.s = self.start_state if options is None else options["state"]
        return self.s, {}

    def step(self, action):
        reward = self.reward(self.s, action)
        self.s = self.nominal_next_state(self.s, action)
        return self.s, reward, True, False, {}


class TransformerDataTests(unittest.TestCase):
    def test_data_contains_numeric_targets_and_deduplicates_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            maps = [ThreeDimensionalMDP(1), ThreeDimensionalMDP(0), ThreeDimensionalMDP(1)]
            generate_q_data(maps, (0, 0.3), (0, np.inf), 0.9, path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["Q"].shape, (8, 2, 6))
                self.assertEqual(len(data["descriptions"]), 2)
                self.assertTrue(np.all(data["bellman_residual"] <= 1e-8))
                self.assertNotIn("task", data.files)
                for Q, index in zip(data["Q"], data["map_index"]):
                    np.testing.assert_array_equal(Q[maps[index].terminal_mask], 0)
                    np.testing.assert_array_equal(Q[~maps[index].terminal_mask], 1)
                train, validation = split_samples(len(data["Q"]))
                self.assertEqual((len(train), len(validation)), (7, 1))
                self.assertFalse(set(train) & set(validation))
                np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(8))
                self.assertTrue(set(data["map_index"][train]) & set(data["map_index"][validation]))

    def test_conditioning_is_in_the_instruction_prompt(self):
        mdp = ThreeDimensionalMDP()
        description = mdp.q_transformer_context()
        prompt = format_prompt(description, 0.3, np.inf)
        self.assertNotEqual(format_prompt(description, 0, 0), prompt)
        self.assertTrue(prompt.startswith("### Instruction:\n"))
        self.assertIn("Predict the Q-table for R = 0.3 and C = infinity.", prompt)
        self.assertIn("Layer z = 0\n0 0", prompt)

    def test_zero_initialization_does_not_import_transformer_dependencies(self):
        script = """
import sys
class BlockTransformerImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('Optional dependency imported by the zero-initialization case')
sys.meta_path.insert(0, BlockTransformerImports())
from robot_2d.robot_2d_motion_mdp import Robot2DMotionMDP
from conformally_robust_controller import ConformallyRobustController
controller = ConformallyRobustController(
    Robot2DMotionMDP(), R=0, C=0, training_episodes=2, horizon=2,
)
assert controller.run_episode()['training']['zero_initialized']
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(HAS_TRANSFORMER, "Install requirement.txt for model checks")
class TransformerModelTests(unittest.TestCase):
    def test_training_checkpoint_reload_and_actual_q_learning_warm_start(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast

        from q_transformer.model import QTransformerPredictor, tokenize_prompts
        from q_transformer.train import train_q_transformer

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            encoder_path = directory / "encoder"
            encoder_path.mkdir()
            vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            vocabulary += ["r", "c", "free", "goal", "infinity", ".", "0", "1", "3", "#", "="]
            (encoder_path / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(encoder_path / "vocab.txt"))
            tokenizer.save_pretrained(encoder_path)
            torch.manual_seed(1)
            config = BertConfig(
                vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=256,
                hidden_dropout_prob=0, attention_probs_dropout_prob=0,
            )
            BertModel(config).save_pretrained(encoder_path)
            data_path = directory / "data.npz"
            generate_q_data(
                [ThreeDimensionalMDP(0), ThreeDimensionalMDP(1)], (0, 0.3), (0, np.inf),
                0.9, data_path,
            )
            checkpoint = directory / "checkpoint"
            history = train_q_transformer(
                data_path, checkpoint, model_name=str(encoder_path), epochs=8,
                batch_size=4, learning_rate=0.01, max_length=256, device="cpu",
            )
            self.assertTrue(np.isfinite(history[-1]["train_loss"]))
            saved = torch.load(checkpoint / "model.pt", weights_only=True)
            self.assertEqual(saved["Q_shape"], (2, 6))
            self.assertNotIn("task", saved)
            self.assertEqual(saved["training"]["epoch"], 8)
            epoch_paths = sorted(checkpoint.glob("checkpoint-*"), key=lambda p: int(p.name[11:]))
            self.assertEqual(len(epoch_paths), 8)
            first_optimizer = torch.load(epoch_paths[0] / "optimizer.pt", weights_only=True)
            last_optimizer = torch.load(epoch_paths[-1] / "optimizer.pt", weights_only=True)
            self.assertAlmostEqual(first_optimizer["param_groups"][0]["lr"], 0.01 * 7 / 8)
            self.assertEqual(last_optimizer["param_groups"][0]["lr"], 0.0)
            self.assertEqual({group["weight_decay"] for group in first_optimizer["param_groups"]},
                             {0.0, 0.01})
            predictor = QTransformerPredictor(checkpoint, device="cpu")
            mdp = ThreeDimensionalMDP()
            Q = predictor(mdp, 0.3, np.inf)
            first_predictor = QTransformerPredictor(epoch_paths[0], device="cpu")
            self.assertFalse(np.array_equal(Q, first_predictor(mdp, 0.3, np.inf)))
            self.assertEqual(Q.shape, (2, 6))
            np.testing.assert_array_equal(Q[mdp.terminal_mask], 0)
            np.testing.assert_array_equal(Q, predictor(mdp, 0.3, np.inf))
            self.assertTrue(all(
                not parameter.requires_grad for parameter in predictor.model.parameters()
            ))
            agent = Tabular_Agent(mdp, 0.9, 0.5, R=0.3, C=np.inf)
            agent.robust_q_learning(2, initial_Q=Q)
            self.assertEqual(agent.evaluation_return[0], Q[0].max())
            self.assertFalse(agent.training_diagnostics["zero_initialized"])
            wrong_shape = ThreeDimensionalMDP()
            wrong_shape.action_space = gym.spaces.Discrete(4)
            with self.assertRaisesRegex(ValueError, "Q-table shape"):
                predictor(wrong_shape, 0.3, np.inf)
            encoded = tokenize_prompts(tokenizer, ["free", "free " * 100], 64)
            self.assertEqual(tuple(encoded["input_ids"].shape), (2, 64))
            self.assertEqual(encoded["attention_mask"][1].sum(), 64)
            self.assertLess(encoded["attention_mask"][0].sum(), 64)


if __name__ == "__main__":
    unittest.main()
