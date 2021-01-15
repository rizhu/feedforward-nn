from collections import defaultdict

import os
import sys

import tensorflow as tf
import numpy as np
import pickle

import ffnn
import loss_functions as lf
import activation_functions as af

CWD = os.getcwd()
NN_PATH = os.path.join(CWD, "saved-neural-nets")
HELP = os.path.join(CWD, "help.txt")

if not os.path.isdir(NN_PATH):
    os.mkdir(NN_PATH)

print("Loading MNIST dataset...")

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

print("MNIST dataset loaded")

x_train_mnist = x_train_mnist.reshape(len(x_train_mnist), -1) / 255
x_test_mnist = x_test_mnist.reshape(len(x_test_mnist), -1) / 255

def net_exists(name: str):
    return os.path.isfile(os.path.join(NN_PATH, f"{name}.nn"))

def save_net(nn: ffnn.FFNN, name: str):
    with open(os.path.join(NN_PATH, f"{name}.nn"), "wb+") as f:
        pickle.dump(nn, f)

def load_net(name: str):
    with open(os.path.join(NN_PATH, f"{name}.nn"), 'rb') as f:
        nn = pickle.load(f)
    return nn

def delete_net(name: str):
    os.remove(os.path.join(NN_PATH, f"{name}.nn"))

def parse_raw(raw):
    tokens = raw.split()
    if len(tokens) > 0:
        cmd = command_map[tokens[0]]
        if (cmd == -1):
            print(f"Command not found: {tokens[0]}")
            if "exit" in raw.lower():
                print("Use 'exit' to exit")
        else:
            cmd(tokens)

def parse_exit(tokens):
    if len(tokens) == 1:
        sys.exit(0)
    else:
        print(f"exit command takes no additional arguments but found {str(len(tokens) - 1)} arguments")

def parse_help(tokens):
    if not len(tokens) == 1:
        print(f"ls command takes no additional arguments but found {str(len(tokens) - 1)} arguments")
        return
    else:
        with open(HELP, 'r') as f:
            help_text = f.read()
        print(help_text)

def parse_ls(tokens):
    if not len(tokens) == 1:
        print(f"ls command takes no additional arguments but found {str(len(tokens) - 1)} arguments")
        return
    for net in os.listdir(NN_PATH):
        if os.path.isfile(os.path.join(NN_PATH, net)):
            print(net[:-3])

def parse_create(tokens):
    if not len(tokens) == 1:
        print(f"create command takes no additional arguments but found {str(len(tokens) - 1)} arguments")
        return

    cmd_name = "create"

    num_hidden_layers = input(f"Number of hidden layers:\n{cmd_name} ❯ ")
    while not (num_hidden_layers.isnumeric() and int(num_hidden_layers) >= 0):
        num_hidden_layers = input(f"Number of hidden layers must be nonnegative integer:\n{cmd_name} ❯ ")

    num_hidden_layers = int(num_hidden_layers)

    layer_dims = input(f"Size of each layer (input as space-separated integers e.g. '10 20 10'):\n{cmd_name} ❯ ")
    layer_tokens = layer_dims.split()
    while not (len(layer_tokens) == num_hidden_layers + 2 and all([t.isnumeric() for t in layer_tokens]) and all([int(t) > 0 for t in layer_tokens])):
        layer_dims = input(f"Number of inputs must equal number of hidden layers + 2 and all inputs must be nonnegative integers:\n{cmd_name} ❯ ")
        layer_tokens = layer_dims.split()

    layer_dims = [int(t) for t in layer_tokens]

    activations = input(f"Activations [identity/i/relu/sigmoid/softmax] of each layer (input as space-separated words e.g. 'relu softmax i'):\n{cmd_name} ❯ ")
    activations_tokens = activations.split()
    while not (len(activations_tokens) == num_hidden_layers + 1 and all([a in af.activation for a in activations_tokens])):
        activations = input(f"Number of activations must be equal to number of hidden layers + 1 and must be one of [identity/i/relu/sigmoid/softmax]:\n{cmd_name} ❯ ")
        activations_tokens = activations.split()

    name = input(f"Name of this neural network:\n{cmd_name} ❯ ")
    while net_exists(name):
        name = input(f"{name} is already in use:\n{cmd_name} ❯ ")

    save_net(ffnn.FFNN(num_hidden_layers, layer_dims, activations_tokens), name)

    print(f"{name} has been successfully created")

def parse_delete(tokens):
    if len(tokens) > 2:
        print(f"delete command takes at most 1 additional argument but found {str(len(tokens) - 1)} arguments")
        return

    if len([net for net in os.listdir(NN_PATH) if os.path.isfile(os.path.join(NN_PATH, net))]) <= 0:
        print("No neural networks to delete")
        return

    cmd_name = "delete"

    if len(tokens) == 2:
        name = tokens[1]
    else:
        name = input(f"Name of neural network to delete:\n{cmd_name} ❯ ")

    while not net_exists(name):
        print(f"{name} is not a neural network. Here is a list of all stored neural nets:")
        for net in os.listdir(NN_PATH):
            if os.path.isfile(os.path.join(NN_PATH, net)):
                print(net[:-3])
        name = input(f"{cmd_name} ❯ ")

    confirm = input(f"Are you sure you want to delete {name} forever? [Y/N]\n{cmd_name} ❯ ")
    while not (confirm == 'y' or confirm == 'n'):
        confirm = input(f"{cmd_name} ❯ ")

    if confirm == 'y':
        delete_net(name)
        print(f"{name} has been successfully deleted")
    else:
        print(f"{name} was not deleted")

def parse_train_mnist(tokens):
    if len(tokens) > 2:
        print(f"train_mnist command takes at most 1 additional argument but found {str(len(tokens) - 1)} arguments")
        return

    cmd_name = "train_mnist"

    name = input(f"Name of neural network to train:\n{cmd_name} ❯ ")
    while not net_exists(name):
        print(f"{name} is not a neural network. Here is a list of all stored neural nets:")
        for net in os.listdir(NN_PATH):
            if os.path.isfile(os.path.join(NN_PATH, net)):
                print(net[:-3])
        name = input(f"{cmd_name} ❯ ")

    nn = load_net(name)

    if len(tokens) == 2:
        if tokens[1] == "-a":
            correct = 0

            for i in range(60000):
                if np.argmax(nn.predict(x_train_mnist[i])) == y_train_mnist[i]:
                    correct += 1

            print(f"Correctly predicted {correct} out of 60000 training examples.")
            print(f"Achieved training accuracy of {(correct / 60000 * 100):.4f}%.")
        else:
            print(f"Second argument should be '-a' to run on all training examples but found {tokens[1]}")
        return

    tr = input(f"Number of training examples:\n{cmd_name} ❯ ")
    while not (tr.isnumeric() and int(tr) > 0 and int(tr) < 60000):
        tr = input(f"Number of training examples must be integer in (0, 60000):\n{cmd_name} ❯ ")
    tr = int(tr)

    cv = input(f"Number of cross-validation examples:\n{cmd_name} ❯ ")
    while not (cv.isnumeric() and int(cv) > 0 and tr + int(cv) <= 60000):
        cv = input(f"Number of cross-validation examples must be positive integer less than or equal to 60000 - number of training examples:\n{cmd_name} ❯ ")
    cv = int(cv)

    x_tr, y_tr = [], []
    x_cv, y_cv = [], []

    correct = 0

    for i in range(tr):
        x_tr.append(x_train_mnist[i])
        d = y_train_mnist[i]
        y_tr.append(np.zeros(10))
        y_tr[i][d] = 1

        if np.argmax(nn.predict(x_tr[i])) == d:
            correct += 1

    print(f"Initial accuracy on training: {(correct / tr * 100):.4f}%")

    correct = 0

    for i in range(tr, tr + cv):
        x_cv.append(x_train_mnist[i])
        d = y_train_mnist[i]
        y_cv.append(np.zeros(10))
        y_cv[i - tr][d] = 1

        if np.argmax(nn.predict(x_cv[i - tr])) == d:
            correct += 1

    print(f"Initial accuracy on cross-validation: {(correct / cv * 100):.4f}%")

    if nn.ran_on_test:
        print(f"{name} has been run on MNIST test data and can no longer be further modified")
        return

    reset_weights = input(f"Reset weights? [Y/N]\n{cmd_name} ❯ ")
    while not (reset_weights.lower() == 'y' or reset_weights.lower() == 'n'):
        reset_weights = input(f"{cmd_name} ❯ ")

    if reset_weights == 'y':
        nn.reset_weights()

    should_train = input(f"Train this neural network? [Y/N]\n{cmd_name} ❯ ")
    while not (should_train.lower() == 'y' or should_train.lower() == 'n'):
        should_train = input(f"{cmd_name} ❯ ")

    if should_train == 'n':
        return

    learning_rate = input(f"Learning rate:\n{cmd_name} ❯ ")
    while True:
        try:
            learning_rate = float(learning_rate)
            if learning_rate > 0:
                break
            raise ValueError
        except ValueError:
            learning_rate = input(f"Learning rate must be positive float:\n{cmd_name} ❯ ")

    iterations = input(f"Number of iterations of stochastic gradient descent:\n{cmd_name} ❯ ")
    while True:
        try:
            iterations = int(iterations)
            if iterations > 0:
                break
            raise ValueError
        except ValueError:
            iterations = input(f"Number of iterations must be positive integer:\n{cmd_name} ❯ ")

    loss_f = input(f"Loss function ['mse'/'ce']\n{cmd_name} ❯ ")
    while not (loss_f.lower() == 'mse' or loss_f.lower() == 'ce'):
        loss_f = input(f"{cmd_name} ❯ ")

    loss_f = loss_f.lower()

    error_code = nn.train(x_tr, y_tr, learning_rate, loss_f, iterations)

    if error_code == -1:
        print(f"Neural network {name} has already been run on test data and can no longer be trained.")
        return
    elif error_code == -2:
        print(f"Neural network {name} does not take in same input size as data.")
        return

    correct = 0

    for i in range(len(y_tr)):
        if np.argmax(nn.predict(x_tr[i])) == np.argmax(y_tr[i]):
            correct += 1

    print(f"Correctly predicted {correct} out of {len(y_tr)} training examples.")
    print(f"Achieved training accuracy of {(correct / len(y_tr) * 100):.4f}%.")

    correct = 0

    for i in range(len(y_cv)):
        if np.argmax(nn.predict(x_cv[i])) == np.argmax(y_cv[i]):
            correct += 1

    print(f"Correctly predicted {correct} out of {len(y_cv)} cross-validation examples.")
    print(f"Achieved cross-validation accuracy of {(correct / len(y_cv) * 100):.4f}%.")

    should_save = input(f"Save this neural network? [Y/N]\n{cmd_name} ❯ ")

    while not (should_save.lower() == 'y' or should_save.lower() == 'n'):
        should_save = input(f"{cmd_name} ❯ ")

    if should_save == 'y':
        save_net(nn, name)
        del nn
        print(f"{name} has been successfully saved")

def parse_test_mnist(tokens):
    if len(tokens) > 2:
        print(f"test_mnist command takes at most 1 additional argument but found {str(len(tokens) - 1)} arguments")
        return

    cmd_name = "test_mnist"
    
    if len(tokens) == 2:
        name = tokens[1]
    else:
        name = input(f"Name of neural network to test:\n{cmd_name} ❯ ")

    while not net_exists(name):
        print(f"{name} is not a neural network. Here is a list of all stored neural nets:")
        for net in os.listdir(NN_PATH):
            if os.path.isfile(os.path.join(NN_PATH, net)):
                print(net[:-3])
        name = input(f"{cmd_name} ❯ ")

    confirm = input(f"Are you sure you want to run {name} on test data? Once {name} has been run on test data, it can no longer be trained. [Y/N]\n{cmd_name} ❯ ")
    while not (confirm.lower() == 'y' or confirm.lower == 'n'):
        confirm = input(f"{cmd_name} ❯ ")

    if confirm == 'n':
        print(f"{name} will not be run on test data.")
        return

    nn = load_net(name)

    error_code = nn.test_mnist(x_test_mnist, y_test_mnist)

    if error_code == -1:
        print(f"Testing failed.")
    else:
        save_net(nn, name)
        del nn

command_map = defaultdict(lambda: -1)
    
command_map["exit"] = parse_exit
command_map["help"] = parse_help
command_map["create"] = parse_create
command_map["delete"] = parse_delete
command_map["train-mnist"] = parse_train_mnist
command_map["test-mnist"] = parse_test_mnist
command_map["ls"] = parse_ls

def main():
    while True:
        raw = input("riznets ❯ ")
        parse_raw(raw)

if __name__ == "__main__":
    main()
