import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

k_vals = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]
train_acc_list_0 = []
train_acc_list_1 = []
test_acc_list_0 = []
test_acc_list_1 = []

os.makedirs("plots", exist_ok=True)

for k in k_vals:
    n = np.load("logfiles/train_acc_k_{}_strategy_1.npy".format(k)) * 100
    train_acc_list_1.append(n[-1])
    plt.plot(n, label="k = {}".format(k))
plt.legend()
plt.title("Train Accuracy Weight Pruning")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("./plots/Train_Accuracy_Weight_Pruning.png")
plt.close()


for k in k_vals:
    n = np.load("logfiles/test_acc_k_{}_strategy_1.npy".format(k)) * 100
    test_acc_list_1.append(n[-1])
    plt.plot(n, label="k = {}".format(k))
plt.legend()
plt.title("Test Accuracy Weight Pruning")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("./plots/Test_Accuracy_Weight_Pruning.png")
plt.close()


for k in k_vals:
    n = np.load("logfiles/train_acc_k_{}_strategy_0.npy".format(k)) * 100
    train_acc_list_0.append(n[-1])
    plt.plot(n, label="k = {}".format(k))
plt.legend()
plt.title("Train Accuracy Neuron Pruning")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("./plots/Train_Accuracy_Neuron_Pruning.png")
plt.close()


for k in k_vals:
    n = np.load("logfiles/test_acc_k_{}_strategy_0.npy".format(k)) * 100
    test_acc_list_0.append(n[-1])
    plt.plot(n, label="k = {}".format(k))
plt.legend()
plt.title("Test Accuracy Neuron Pruning")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("./plots/Test_Accuracy_Neuron_Pruning.png")
plt.close()


plt.plot(k_vals, train_acc_list_0, label="Training Accuracy")
plt.plot(k_vals, test_acc_list_0, label="Testing Accuracy")
plt.title("Accuracy vs Sparsity for Neuron Pruning")
plt.legend()
plt.xlabel("Sparsity Percent")
plt.ylabel("Accuracy")
plt.savefig("./plots/Accuracy_vs_Sparsity_for_Neuron_Pruning.png")
plt.close()


plt.plot(k_vals, train_acc_list_1, label="Training Accuracy")
plt.plot(k_vals, test_acc_list_1, label="Testing Accuracy")
plt.title("Accuracy vs Sparsity for Weight Pruning")
plt.legend()
plt.xlabel("Sparsity Percent")
plt.ylabel("Accuracy")
plt.savefig("./plots/Accuracy_vs_Sparsity_for_Weight_Pruning.png")
plt.close()


plt.plot(k_vals, train_acc_list_1, label="Weight Pruning")
plt.plot(k_vals, train_acc_list_0, label="Neuron Pruning")
plt.title("Training Accuracy Comparison for different Pruning Strategies")
plt.legend()
plt.xlabel("Sparsity Percent")
plt.ylabel("Accuracy")
plt.savefig("./plots/Training_Accuracy_Comparison_for_different_Pruning_Strategies.png")
plt.close()


plt.plot(k_vals, test_acc_list_1, label="Weight Pruning")
plt.plot(k_vals, test_acc_list_0, label="Neuron Pruning")
plt.title("Testing Accuracy Comparison for different Pruning Strategies")
plt.legend()
plt.xlabel("Sparsity Percent")
plt.ylabel("Accuracy")
plt.savefig("./plots/Testing_Accuracy_Comparison_for_different_Pruning_Strategies.png")
plt.close()

