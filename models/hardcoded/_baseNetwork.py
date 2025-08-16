import numpy as np

class _baseNetwork:
    
    def __init__(self, input_size=28*28, num_classes=10):
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass
    
    def forward(self):
        pass
    
    def softmax(self, scores):
        # Softmax x_i = e^(x_i) / sum_j e^(x_j)
        numerator = np.exp(scores - np.max(scores, axis=1, keepdms=True))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator
    
    def cross_entropy_loss(self, y_prediction, y):
        probabilities = np.clip(y_prediction, 1e-12, 1)
        log_values = np.log(probabilities[np.arange(len(y)), y])
        return -np.mean(log_values)

    def compute_accuracy(self, y_prediction, y):
        prediction_x = np.argmax(y_prediction, axis=1)
        return np.mean(prediction_x == y)
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def sigmoid_derivative(self, X):
        sigmoid = self.sigmoid(X)
        return sigmoid * (1 - sigmoid)

    def ReLU(self, X):
        return np.maximum(0, X)
    
    def ReLU_derivative(self, X):
        # ReLU max(x, 0) if x > 0, x, else 0
        # Derivative of ReLU -> if x > 0, 1, else 0
        return np.where(X > 0, 1, 0)