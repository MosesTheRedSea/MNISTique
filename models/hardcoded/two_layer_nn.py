import numpy as np
np.random.seed(1024)
from _baseNetwork import _baseNetwork

class TwoLayerNeuralNet(_baseNetwork):

    def __init__(self, input_size=28*28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)
        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):

        # initialize weights of the network
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)

        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)

        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize weights
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        
        loss = None
        accuracy = None

        # Linear Model = F(x, W) = x*W + b
        self.N = X.shape[0] # batch size of the images
        self.X = X 
        
        # Layer Linar 1
        linear_layer1 = np.dot(X, self.weights["W1"]) + self.weights["b1"]
        self.linear_layer1_output = linear_layer1

        # Sigmoid Layer | Activation Layer
        sigmoid_layer = self.sigmoid(self.linear_layer1_output)
        self.activation = sigmoid_layer

        # Layer Linear 2
        linear_layer2 = np.dot(self.activation, self.weights["W2"]) + self.weights["b2"]
        self.linear_layer2_output = linear_layer2

        # Softmax Layer | Converts the outputs that we got to a probability from [0, 1]
        softmax_layer = self.softmax(self.linear_layer2_output)
        self.probabilities = softmax_layer

        # Cross Entropy Loss
        loss = self.cross_entropy_loss(self.probabilities, y)

        # Accuracy
        accuracy = self.compute_accuracy(self.probabilities, y)
        if mode != 'train':
            return loss, accuracy

        derivative_loss_respect_softmax = self.probabilities.copy()
        derivative_loss_respect_softmax[np.arange(self.N), y] -= 1
        derivative_loss_respect_softmax /= self.N

        # Compute Gradients For W2, b2 first
        self.gradients["W2"] = np.dot(self.activation.T, derivative_loss_respect_softmax)
        self.gradients["b2"] = np.sum(derivative_loss_respect_softmax, axis=0)
        
        # Analystical derivative of sigmoid function in self.sigmoid_dev first
        hidden = np.dot(derivative_loss_respect_softmax, self.weights["W2"].T)
        derivative_loss_respect_sigmoid = hidden * self.sigmoid_dev(self.linear_layer1_output)

        # Compute Gradients for W1, b1
        self.gradients["W1"] = np.dot(self.X.T, derivative_loss_respect_sigmoid)
        self.gradients["b1"] = np.sum(derivative_loss_respect_sigmoid, axis=0)
       
        return loss, accuracy