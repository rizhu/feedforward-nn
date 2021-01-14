import ffnn
import tensorflow as tf
import numpy as np
import os
import pickle

import loss_functions as lf

CWD = os.getcwd()
NN_PATH = os.path.join(CWD, "saved-neural-nets")

if not os.path.isdir(NN_PATH):
    os.mkdir(NN_PATH)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("MNIST loaded")

x_train = x_train.reshape(len(x_train), -1) / 255
x_test = x_test.reshape(len(x_test), -1) / 255

should_load = input("Load an existing neural network? [Y/N]\n> ")

while not (should_load.lower() == 'y' or should_load.lower() == 'n'):
    should_load = input("> ")

if should_load == 'y':
    load_name = input("Name of existing neural net (do not include .nn filename extension)\n> ")
    while not os.path.isfile(os.path.join(NN_PATH, load_name + ".nn")):
        load_name = input("> ")
    with open(os.path.join(NN_PATH, load_name + ".nn"), 'rb') as f:
        nn = pickle.load(f)
else:
    nn = ffnn.FFNN(1, [784, 300, 10], ["relu", "softmax"])

print("NN set up")

x_tr, y_tr = [], []
x_cv, y_cv = [], []

tr = int(input("Training data size: "))
cv = int(input("Cross-validation data size: "))

correct = 0

for i in range(tr):
    x_tr.append(x_train[i])
    d = y_train[i]
    y_tr.append(np.zeros(10))
    y_tr[i][d] = 1

    if np.argmax(nn.predict(x_tr[i])) == d:
        correct += 1

print("Initial accuracy on training: " + str(correct / tr * 100) + "%%")

correct = 0

for i in range(tr, tr + cv):
    x_cv.append(x_train[i])
    d = y_train[i]
    y_cv.append(np.zeros(10))
    y_cv[i - tr][d] = 1

    if np.argmax(nn.predict(x_cv[i - tr])) == d:
        correct += 1

print("Initial accuracy on cross-validation: " + str(correct / cv * 100) + "%%")

should_train = input("Train this neural network? [Y/N]\n> ")
while not (should_train.lower() == 'y' or should_train.lower() == 'n'):
    should_train = input("> ")

if should_train.lower() == 'y':
    rate = float(input("Learning rate: "))
    iterations = int(input("Number of iterations: "))

    print("Beginning training")

    nn.train(x_tr, y_tr, rate, "ce", iterations)

    print("Finished training with learning rate " + str(rate) + " and " + str(iterations) + " iterations.")

    correct = 0

    for i in range(len(y_tr)):
        if np.argmax(nn.predict(x_tr[i])) == np.argmax(y_tr[i]):
            correct += 1

    print("Final accuracy on training: " + str(correct / tr * 100) + "%%")

    correct = 0

    for i in range(len(y_cv)):
        if np.argmax(nn.predict(x_cv[i])) == np.argmax(y_cv[i]):
            correct += 1

    print("Final accuracy on cross-validation: " + str(correct / cv * 100) + "%%")

    should_save = input("Save this neural network? [Y/N]\n> ")

    while not (should_save.lower() == 'y' or should_save.lower() == 'n'):
        should_save = input("> ")

    if should_save.lower() == 'y':
        nn_name = input("Name of this neural network:\n> ")
        nn_name += ".nn"
        with open(os.path.join(NN_PATH, nn_name), "wb+") as f:
            pickle.dump(nn, f)
