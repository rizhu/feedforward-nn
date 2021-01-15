import numpy as np

import activation_functions as af
import loss_functions as lf

class FFNN:

    """
    Square root of 2 constant
    """
    sqrt_2 = np.sqrt(2)

    def __init__(self, num_hidden_layers: int, layer_dims: list, activations: list):
        """
        num_hidden_layers:  non-negative integer representing number of hidden layers
        layer_dims:         list of non-negative integers representing size of each hidden layer
        activations:        list of strings for activation functions between layers
        """

        self.weights = [None]
        self.biases = [None]

        self.activations = [None]
        self.activations_d = [None]

        self.ran_on_test = False

        for i in range(1, len(layer_dims)):
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i - 1]) * (FFNN.sqrt_2 / layer_dims[i - 1]))
            self.biases.append(np.zeros(layer_dims[i]))
            self.activations.append(af.activation[activations[i - 1]])
            self.activations_d.append(af.activation_d[activations[i - 1]])

    def train(self, x: list, y: list, learning_rate: int, loss: str, iterations: int):
        """
        x:              list of training inputs
        y:              list of training labels
        learning_rate:  gradient descent step-size
        loss:           loss function to evaluate predictions on
        iterations:     number of iterations to train
        """

        if self.ran_on_test:
            return -1

        if not x[0].shape[0] == self.weights[1].shape[1]:
            return -2

        print("Beginning training")

        z = []
        a = []
        delta = [None] * len(self.weights)

        J_d = lf.loss_d[loss]

        curr_loss = 0
        prev_loss = 0

        for curr_iteration in range(iterations):
            for j in range(len(x)):
                z.append(None)
                a.append(x[j])
                for i in range(1, len(self.weights)):
                    z.append(self.weights[i] @ a[i - 1] + self.biases[i])
                    a.append(self.activations[i](z[i]))

                delta[-1] = J_d(y[j], a[-1]) * self.activations_d[-1](z[-1])

                self.weights[-1] -= learning_rate * (delta[-1].reshape(delta[-1].shape[0], 1) @ a[-2].reshape(1, a[-2].shape[0]))
                self.biases[-1] -= learning_rate * delta[-1]

                for i in range(len(a) - 2, 0, -1):
                    delta[i] = (self.weights[i + 1].T @ delta[i + 1].reshape(delta[i + 1].shape[0], 1)).flatten() * self.activations_d[i](z[i])

                    self.weights[i] -= learning_rate * (delta[i].reshape(delta[i].shape[0], 1) @ a[i - 1].reshape(1, a[i - 1].shape[0]))
                    self.biases[i] -= learning_rate * delta[i]

                z.clear()
                a.clear()

            if curr_iteration > 0 and curr_iteration % (iterations / 20) == 0:
                print(str(int(curr_iteration / iterations * 100)) + "%% complete...")

        print("Finished training")

    def predict(self, x):
        """
        x:  input to predict
        """

        res = x
        for i in range(1, len(self.weights)):
            res = self.activations[i](self.weights[i] @ res + self.biases[i])
        return res

    def reset_weights(self):
        if self.ran_on_test:
            return -1

        for i in range(1, len(self.weights)):
            self.weights[i] = np.random.randn(self.weights[i].shape[0], self.weights[i].shape[1]) * (FFNN.sqrt_2 / self.weights[i].shape[1])
            self.biases[i] = np.zeros(self.biases[i].shape[0])

    def test(self, x_test: list, y_test: list):
        """
        x_test: list of test inputs
        y_test: list of test labels
        """

        if not len(x_test) == len(y_test):
            print(f"Dimension mismatch between test inputs and labels. Found {len(x_test)} test inputs and {len(y_test)} labels")
            return

        correct = 0

        for i in range(len(x_test)):
            if np.argmax(self.predict(x_test[i])) == np.argmax(y_test[i]):
                correct += 1

        self.ran_on_test = True

        print(f"Correctly predicted {correct} out of {len(y_test)} test examples.")
        print(f"Achieved test accuracy of {(correct / len(y_test)):.4f}%.")
