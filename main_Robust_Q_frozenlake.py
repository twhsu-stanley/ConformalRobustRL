import numpy as np
import matplotlib.pyplot as plt
import pickle
import gymnasium as gym
from Tabular_Agent import Tabular_Agent
from utils_tabular import plot_frozenlake_tabular, plot_evaluation_return

"""Robust Q-learning control"""

if __name__ == "__main__":
    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], 
                   max_episode_steps = 100, map_name="4x4", is_slippery=is_slippery)
    gamma = 0.95

    # Train Robust Q
    n_episodes = 4000
    R = 0.2 # Contamination probability for the R-C uncertainty set
    C = 0.0 # Grid-coordinate deviation radius for the R-C uncertainty set
    n_trials = 30
    evaluation_return = []
    for i in range(n_trials):    
        robust_q_agent = Tabular_Agent(env, gamma, lr_init = 0.5, step_start_decay_lr = 10000,
                                       epsilon_init = 1.0, epsilon_lb = 0.1, epsilon_decay_rate = 0.9995,
                                       R = R, C = C)
        robust_q_agent.Robust_Q_learning(n_episodes)
        evaluation_return.append(robust_q_agent.evaluation_return)

        if i == 0:
            plot_frozenlake_tabular(robust_q_agent, is_slippery=is_slippery, algorithm="Robust Q-Learning")
            plot_evaluation_return(evaluation_return)
    plot_evaluation_return(evaluation_return)

    filename = f"Tabular/Robust_Q_frozenlake_R{R}_C{C}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(evaluation_return, f)
    
    ###################################################################################
    # Regular Q for comparison
    n_episodes = 4000  
    q_agent = Tabular_Agent(env, gamma, lr_init = 0.5, step_start_decay_lr = 10000,
                            epsilon_init = 1.0, epsilon_lb = 0.1, epsilon_decay_rate = 0.9995,
                            R = 0)
    q_agent.Q_learning(n_episodes)
    #plot_frozenlake_tabular(q_agent, is_slippery=is_slippery, algorithm="Q-Learning")
    #plot_evaluation_return([q_agent.evaluation_return])

    ###################################################################################
    # Test the policies on uncertain transitions
    p = R
    G_robust = 0
    G = 0
    n_test = 50000
    for t in range(n_test):
        print(f"Episode {t+1}/{n_test}")
        G_robust += robust_q_agent.sim_perturbed(p, C=C)
        G += q_agent.sim_perturbed(p, C=C)
    G_robust = G_robust/n_test
    G = G/n_test

    print(f"Robust Q-agent MC return: G_robust = {G_robust}")
    print(f"Q-agent MC return: G = {G}")
    print(f"G_robust - G = {G_robust - G}")
