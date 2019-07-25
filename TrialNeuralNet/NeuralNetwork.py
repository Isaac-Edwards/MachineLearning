import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x                                                  # Input 
        self.weights1 = np.random.rand(self.input.shape[1],4)           # random 4 weights
        self.weights2 = np.random.rand(4,1)                             # second layer of weights?
        self.y = y                                                      # output layer
        self.output = np.zeros(y.shape)                                 # our guess 

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(x):
        return sigmoid(x)*(1-sigmoid(x))

    def feedforward(self):
        # biases assumed to be 0
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T))

        self.weights1 += d_weights1
        self.weights2 += d_weights2
