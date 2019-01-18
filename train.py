import sys
import argparse
from datetime import datetime
from random import random
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.enable_eager_execution()

tf.set_random_seed(144)

parser = argparse.ArgumentParser(description="Train a Pruned Neural Network")
parser.add_argument("-p", "--pruning_strategy", type=int, default=0,
                    help="Choose the pruning strategy. 1 for weight pruning,\
                          0 for neuron pruning")
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="Total number of epochs to train the model")
parser.add_argument("-k", "--prune_percent", default=0,
                    help="Percentage of the neurons/weights to be pruned")
parser.add_argument("-a", "--train_accuracy_file", default="./train_accuracy.npy",
                    help="Path to store the train accuracy file")
parser.add_argument("-t", "--test_accuracy_file", default="./test_accuracy.npy",
                    help="Path to store the test accuracy file")
parser.add_argument("-l", "--logging_file", default="./logs.txt",
                    help="Path to Logging File")


class SimpleNeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.W1 = tfe.Variable(tf.random_normal((28 * 28, 1000)))
        self.W2 = tfe.Variable(tf.random_normal((1000, 1000)))
        self.W3 = tfe.Variable(tf.random_normal((1000, 500)))
        self.W4 = tfe.Variable(tf.random_normal((500, 200)))
        self.W5 = tfe.Variable(tf.random_normal((200, 10)))

    def call(self, x):
        # Matmul followed by relu forms 1 layer
        x = tf.matmul(x, self.W1)
        x = tf.maximum(x, 0)
        x = tf.matmul(x, self.W2)
        x = tf.maximum(x, 0)
        x = tf.matmul(x, self.W3)
        x = tf.maximum(x, 0)
        x = tf.matmul(x, self.W4)
        x = tf.maximum(x, 0)
        x = tf.matmul(x, self.W5)
        return x


def weight_pruning(x, k):
    count = int(x.shape[0] * x.shape[1])
    mask = np.ones(count, dtype=np.float32)
    count = k * count / 100
    # Create a mask with deletes the weights with minimum absolute value
    mask[tf.math.top_k(-tf.math.abs(tf.reshape(x, [-1])), count)[1]] = 0.0
    return tf.multiply(x, tf.reshape(mask, x.shape))


def neuron_pruning(x, k):
    count = k * int(x.shape[1]) / 100
    mask = np.ones(x.shape, dtype=np.float32)
    # Create a mask to delete entire columns of weight with minimum norm
    mask[:, tf.math.top_k(-tf.norm(x, axis=0), count)[1]] = 0.0
    return tf.multiply(x, mask)


class PruningNeuralNetwork(SimpleNeuralNetwork):
    def __init__(self, k=0.0, pruning=weight_pruning, training=True):
        super(PruningNeuralNetwork, self).__init__()
        self.k = k
        self.pruning = pruning
        self.training = training

    def call(self, x):
        # We don't need to prune at inference time as the weights will not
        # change
        if self.training:
            self.W1 = self.pruning(self.W1, self.k)
            self.W2 = self.pruning(self.W2, self.k)
            self.W3 = self.pruning(self.W3, self.k)
            self.W4 = self.pruning(self.W4, self.k)
        return super(PruningNeuralNetwork, self).call(x)


def loss(model, inputs, labels):
    # Minimize the cross entropy loss
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=model(inputs), labels=labels))


def train_step(loss, net, opt, x, y):
    opt.minimize(lambda: loss(net, x, y),
                 global_step=tf.train.get_or_create_global_step())


data = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels))\
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))\
        .shuffle(buffer_size=1000)

test_ds = tf.data.Dataset.from_tensor_slices((data.test.images, data.test.labels))\
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))\
        .shuffle(buffer_size=1000)

args = parser.parse_args()

EPOCHS = args.epochs
k = float(args.prune_percent)
strategy = neuron_pruning if args.pruning_strategy == 0 else weight_pruning
path = args.checkpoint

sys.stdout = open(args.logging_file, 'w')

train_acc_history = np.zeros(EPOCHS)
test_acc_history = np.zeros(EPOCHS)

train_accuracy = tfe.metrics.Accuracy()
test_accuracy = tfe.metrics.Accuracy()

# Tensorboard logging
# random is needed as we run multiple models parallely, logging of some models
# might fail
summary_writer = tf.contrib.summary.create_file_writer("logs/logs-{}-{}".format(datetime.now().strftime("%Y-%m-%d,%H:%M:%S"),
                                                                                random()),
                                                       flush_millis=10000)
summary_writer.set_as_default()

net = PruningNeuralNetwork(k, strategy)

# We shall use ADAM for all our experiments.
opt = tf.train.AdamOptimizer(0.001)

print("Starting for k = {}".format(k))

for epoch in range(EPOCHS):
    for (xb, yb) in tfe.Iterator(train_ds.batch(256)):
        train_step(loss, net, opt, xb, yb)
        # Compue Training accuracy for the batch. Ideally we should compute it
        # once the epoch is complete but this value would be close to that.
        train_accuracy(tf.argmax(net(tf.constant(xb)), axis=1), tf.argmax(tf.constant(yb), axis=1))

    print("Training Accuracy after {} Epochs = {}".format(epoch, train_accuracy.result().numpy()))

    train_acc_history[epoch] = train_accuracy.result().numpy()

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("Train Accuracy", train_acc_history[epoch])

    # Turn off training for faster inference
    net.training = False
    for (xb, yb) in tfe.Iterator(test_ds.batch(256)):
        # Compute the Test Accuracy
        test_accuracy(tf.argmax(net(tf.constant(xb)), axis=1), tf.argmax(tf.constant(yb), axis=1))
    net.training = True

    print("Testing Accuracy after {} Epochs = {}".format(epoch, test_accuracy.result().numpy()))

    test_acc_history[epoch] = test_accuracy.result().numpy()

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("Test Accuracy", test_acc_history[epoch])

np.save(args.train_accuracy_file, train_acc_history)
np.save(args.test_accuracy_file, test_acc_history)

