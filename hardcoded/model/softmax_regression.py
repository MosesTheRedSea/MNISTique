import numpy as np

from ._baseNetwork import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__self(self, input_size=28*28, num_classes=10):
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        np.random.seed(1024)

        self.weights['W1'] = 0.001 * np.randn(self.input_size, self.num_classes)
        self.gradient['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):

        # Forward Pass
        self.N = X.shape[0]
        self.X = X

        # Linear Layer
        self.linear_layer_output = np.dot(X, self.weights["W1"])

        # ReLU Activation Layer
        self.activation = self.ReLU(self.linear_layer_output)

        # Softmax Output | Outputs -> Probabilities [0, 1]
        self.probabilities = self.softmax(self.activation)

        # Cross Entropy | Loss
        loss = self.cross_entropy_loss(self.probabilities, y)
        accuracy = self.compute_accuracy(self.probabilities, y)

        if mode != 'train':
            return loss, accuracy
        
        # Backward Pass
        ds = self.probabilities.copy()
        ds[np.arange(self.N), y] -= 1
        ds /= self.N

        dr = ds * self.ReLU_derivative(self.linear_layer_output)

        dw = np.dot(X.T, dr)

        self.gradients['W1'] = dw

        return loss, accuracy

        
