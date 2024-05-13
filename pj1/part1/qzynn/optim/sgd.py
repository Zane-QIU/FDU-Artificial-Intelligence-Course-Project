import os
import sys
import numpy as np
from .base import Optimizer

sys.path.append(os.getcwd())


class SGD(Optimizer):
    def __init__(self, model, lr=0.0):
        """
        Initializes the SGD optimizer.

        Args:
            model: The model to be optimized.
            lr: The learning rate for the optimizer.
        """
        self.model = model
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step.
        """
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.lr * layer.grads[key]

    def update_learning_rate(self, new_lr):
        """
        Updates the learning rate.

        Args:
            new_lr: The new learning rate.
        """
        self.lr = new_lr
