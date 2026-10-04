import numpy as np
import pickle
import gymnasium as gym
from Tabular_Agent import Tabular_Agent
from utils_tabular import plot_frozenlake_tabular

"""Off-policy TD(0) Q-learning control"""

if __name__ == "__main__":
    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False#True#
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)

    gamma = 0.95
    n_episodes = 80000
    learning_rate_init = 0.1

    q_agent = Tabular_Agent(env, gamma, learning_rate_init)
    q_agent.Q_learning(n_episodes)
    plot_frozenlake_tabular(q_agent, is_slippery=is_slippery, algorithm="Q-Learning")

    filename = f"Q_learning_slippery_{is_slippery}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(q_agent.evaluation_return, f)
