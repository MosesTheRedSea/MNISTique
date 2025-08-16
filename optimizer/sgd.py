from ._base_optimizer import _BaseOptimizer
import numpy as np

class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):

        self.apply_regularization(model)

        for weight in model.weights:
            model.weights[weight] -= self.learning_rate * model.gradients[weight]