import matplotlib.pyplot as plt
import pickle
import numpy as np

# Compare the evaluation returns from different algorithms
# is_slippery = False
with open("MC_slippery_False.pkl", "rb") as f:
    mc_evaluation_return = pickle.load(f)

with open("SARSA_slippery_False.pkl", "rb") as f:
    sarsa_evaluation_return = pickle.load(f)

with open("Q_learning_slippery_False.pkl", "rb") as f:
    q_learning_evaluation_return = pickle.load(f)

plt.figure()
plt.plot(mc_evaluation_return, label="MC Control")
plt.plot(sarsa_evaluation_return, label="SARSA")
plt.plot(q_learning_evaluation_return, label="Q-Learning")
plt.hlines(0.774, 0, max(len(mc_evaluation_return), len(sarsa_evaluation_return), len(q_learning_evaluation_return)), colors='k', linestyles='dashed', label="Optimal Return by Value Iteration")
plt.grid()
plt.xlabel("Culmulative Time Steps")
plt.ylabel("Evaluation Return")
plt.legend()
plt.title("is_slippery = False")
plt.show()

# is_slippery = True
with open("MC_slippery_True.pkl", "rb") as f:
    mc_evaluation_return = pickle.load(f) 

with open("SARSA_slippery_True.pkl", "rb") as f:
    sarsa_evaluation_return = pickle.load(f)

with open("Q_learning_slippery_True.pkl", "rb") as f:
    q_learning_evaluation_return = pickle.load(f)

plt.figure()
plt.plot(mc_evaluation_return, label="MC Control")
plt.plot(sarsa_evaluation_return, label="SARSA")
plt.plot(q_learning_evaluation_return, label="Q-Learning")
plt.hlines(0.18, 0, max(len(mc_evaluation_return), len(sarsa_evaluation_return), len(q_learning_evaluation_return)), colors='k', linestyles='dashed', label="Optimal Return by Value Iteration")
plt.grid()
plt.xlabel("Culmulative Time Steps")
plt.ylabel("Evaluation Return")
plt.title("is_slippery = True")
plt.legend()
plt.show()