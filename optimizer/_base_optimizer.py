class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
         for weight in model.weights:
            if "W" in weight:
                model.gradients[weight] += self.reg * model.weights[weight]