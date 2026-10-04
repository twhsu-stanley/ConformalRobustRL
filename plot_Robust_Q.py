import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils_tabular import calc_evaluation_return_mean_std

# Plot 1: fix C = 1.0, compare different R values

with open("Tabular/Robust_Q_frozenlake_R0.0_C1.0.pkl", "rb") as f:
    evaluation_return_R0_C1 = pickle.load(f)
mean_R0_C1, std_R0_C1, _ = calc_evaluation_return_mean_std(evaluation_return_R0_C1)

with open("Tabular/Robust_Q_frozenlake_R0.1_C1.0.pkl", "rb") as f:
    evaluation_return_R01_C1 = pickle.load(f)
mean_R01_C1, std_R01_C1, _ = calc_evaluation_return_mean_std(evaluation_return_R01_C1)

with open("Tabular/Robust_Q_frozenlake_R0.2_C1.0.pkl", "rb") as f:
    evaluation_return_R02_C1 = pickle.load(f)
mean_R02_C1, std_R02_C1, _ = calc_evaluation_return_mean_std(evaluation_return_R02_C1)

with open("Tabular/Robust_Q_frozenlake_R0.4_C1.0.pkl", "rb") as f:
    evaluation_return_R04_C1 = pickle.load(f)
mean_R04_C1, std_R04_C1, _ = calc_evaluation_return_mean_std(evaluation_return_R04_C1)

plt.figure()
plt.plot(mean_R0_C1, label="R = 0, C = 1.0")
plt.fill_between(range(len(mean_R0_C1)), mean_R0_C1 - std_R0_C1, mean_R0_C1 + std_R0_C1, alpha=0.2)

plt.plot(mean_R01_C1, label="R = 0.1, C = 1.0")
plt.fill_between(range(len(mean_R01_C1)), mean_R01_C1 - std_R01_C1, mean_R01_C1 + std_R01_C1, alpha=0.2)

plt.plot(mean_R02_C1, label="R = 0.2, C = 1.0")
plt.fill_between(range(len(mean_R02_C1)), mean_R02_C1 - std_R02_C1, mean_R02_C1 + std_R02_C1, alpha=0.2)

plt.plot(mean_R04_C1, label="R = 0.4, C = 1.0")
plt.fill_between(range(len(mean_R04_C1)), mean_R04_C1 - std_R04_C1, mean_R04_C1 + std_R04_C1, alpha=0.2)

plt.grid()
plt.xlim(0, 30000)
plt.ylim(0, 0.8)
plt.xlabel("Cumulative Time Steps")
plt.ylabel("Evaluation Return: V(initial state)")
plt.title("Robust Q-Learning on FrozenLake: Fixed C = 1.0")
plt.legend(loc="upper right")
plt.savefig("Tabular/Robust_Q_frozenlake_fixed_C1.0_compare_R.png", dpi=300, bbox_inches="tight")


# Plot 2: fix R = 0.2, compare different C values

with open("Tabular/Robust_Q_frozenlake_R0.2_C0.0.pkl", "rb") as f:
    evaluation_return_R02_C0 = pickle.load(f)
mean_R02_C0, std_R02_C0, _ = calc_evaluation_return_mean_std(evaluation_return_R02_C0)

with open("Tabular/Robust_Q_frozenlake_R0.2_C1.0.pkl", "rb") as f:
    evaluation_return_R02_C1 = pickle.load(f)
mean_R02_C1, std_R02_C1, _ = calc_evaluation_return_mean_std(evaluation_return_R02_C1)

with open("Tabular/Robust_Q_frozenlake_R0.2_C1.5.pkl", "rb") as f:
    evaluation_return_R02_C15 = pickle.load(f)
mean_R02_C15, std_R02_C15, _ = calc_evaluation_return_mean_std(evaluation_return_R02_C15)

with open("Tabular/Robust_Q_frozenlake_R0.2_C2.0.pkl", "rb") as f:
    evaluation_return_R02_C2 = pickle.load(f)
mean_R02_C2, std_R02_C2, _ = calc_evaluation_return_mean_std(evaluation_return_R02_C2)

plt.figure()
plt.plot(mean_R02_C0, label="R = 0.2, C = 0")
plt.fill_between(range(len(mean_R02_C0)), mean_R02_C0 - std_R02_C0, mean_R02_C0 + std_R02_C0, alpha=0.2)

plt.plot(mean_R02_C1, label="R = 0.2, C = 1")
plt.fill_between(range(len(mean_R02_C1)), mean_R02_C1 - std_R02_C1, mean_R02_C1 + std_R02_C1, alpha=0.2)

#plt.plot(mean_R02_C15, label="R = 0.2, C = 1.5")
#plt.fill_between(range(len(mean_R02_C15)), mean_R02_C15 - std_R02_C15, mean_R02_C15 + std_R02_C15, alpha=0.2)

plt.plot(mean_R02_C2, label="R = 0.2, C = 2")
plt.fill_between(range(len(mean_R02_C2)), mean_R02_C2 - std_R02_C2, mean_R02_C2 + std_R02_C2, alpha=0.2)

plt.grid()
plt.xlim(0, 30000)
plt.ylim(0, 0.8)
plt.xlabel("Cumulative Time Steps")
plt.ylabel("Evaluation Return: V(initial state)")
plt.title("Robust Q-Learning on FrozenLake: Fixed R = 0.2")
plt.legend(loc="upper right")
plt.savefig("Tabular/Robust_Q_frozenlake_fixed_R0.2_compare_C.png", dpi=300, bbox_inches="tight")

plt.show()