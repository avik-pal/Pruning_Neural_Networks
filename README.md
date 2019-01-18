# Pruning Challenge

## Problem Description

We have a simple 3 hidden layered neural network. All the layers are a simple linear layer without any bias, i.e., a simple matmul.

$$ out = relu(input \times W)$$

The entire neural network is simply described as:

$$ out_1 = relu(input \times W_1) $$
$$ out_2 = relu(out_1 \times W_2) $$
$$ out_3 = relu(out_2 \times W_3) $$
$$ out_4 = relu(out_3 \times W_4) $$
$$ out = softmax(out_4 \times W_5) $$

Now we have 2 pruning strategies -
1. Weight Pruning
2. Neuron/Unit Pruning

We need to compare the two strategies based on the accuracy attained while training the model on MNIST.

## Steps to Reproduce Results

```bash
# Create a virtual environment
python3 -m virtualenv pruning
# Activate the environment
source pruning/bin/activate
# Install the dependencies
pip install -r requirements.txt
```

The pruning step is expensive and the models are trained for `100` epochs, so it is recommended to carry out the training in a GPU/TPU environment.

```bash
# Train the models
python main.py
# To get the plots
python plotting.py
```

Don't alter the directories generated by `main.py`, `plotting.py` needs them in the exact order.

**NOTE**: We train 2 models at a time. So it is recommended to have sufficient GPU memory.

You can use `train.py` to train individual models. For the instructions use `python train.py --help`. This will list the arguments it needs.

If you want to train 1 model at a time use the `main_alternate.py` script.

## Results

### Accuracy vs Epoch Plots

| ![Training Neuron Pruning][./plots/Train_Accuracy_Neuron_Pruning.png] | ![Training Weight Pruning][./plots/Train_Accuracy_Weight_Pruning.png] |
|:-:|:-:|
| ![Testing Neuron Pruning][./plots/Test_Accuracy_Neuron_Pruning.png] | ![Testing Weight Pruning][./plots/Test_Accuracy_Weight_Pruning.png] |

### Accuracy vs Sparsity

| ![Accuracy Neuron Pruning][./plots/Accuracy_vs_Sparsity_for_Neuron_Pruning.png] | ![Accuracy Weight Pruning][./plots/Accuracy_vs_Sparsity_for_Weight_Pruning.png] |
|:-:|:-:|
| ![Training Accuracy Pruning]["./plots/Training_Accuracy_Comparison_for_different_Pruning_Strategies.png"] | ![Testing Accuracy Pruning]["./plots/Testing_Accuracy_Comparison_for_different_Pruning_Strategies.png"] |

## Hypothesis

## Tricks to Speed up the Computation with Sparse Models
