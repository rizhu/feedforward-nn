import numpy as np
import activation_functions as af
import loss_functions as lf

class FFNN:

    """
    Small constant used to avoid dropout on weight initialization and ensure well-defined inputs
    to log functions
    """
    tiny_const = 0.0001

    def __init__(self, num_hidden_layers: int, layer_dims: list, activations: list):
        """
        num_hidden_layers:  non-negative integer representing number of hidden layers
        layer_dims:         list of non-negative integers representing size of each hidden layer
        activations:        list of strings for activation functions between layers
        """
        assert len(layer_dims) == num_hidden_layers + 2
        assert len(activations) == len(layer_dims) - 1

        self.weights = [None]
        self.biases = [None]

        self.activations = [None]
        self.activations_d = [None]

        for i in range(1, len(layer_dims)):
            self.weights.append(np.random.rand(layer_dims[i], layer_dims[i - 1]) * 2 - 1)
            self.biases.append(np.random.rand(layer_dims[i]) + FFNN.tiny_const * 2 - 1)
            self.activations.append(af.activation[activations[i - 1]])
            self.activations_d.append(af.activation_d[activations[i - 1]])

    def train(self, x, y, learning_rate: int, loss: str, iterations: int):
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
                print(str(int(curr_iteration / iterations * 100)) + "%% complete")

    def predict(self, x):
        res = x
        for i in range(1, len(self.weights)):
            res = self.activations[i](self.weights[i] @ res + self.biases[i])
        return res