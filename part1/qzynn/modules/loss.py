import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Loss

class CrossEntropyLoss:
    """
    CrossEntropyLoss is a class that computes the cross-entropy loss for a given input and target.

    Args:
        model (object): The model object.
        label_num (int): The number of labels.
        reduction (str, optional): Specifies the reduction to apply to the output. Default is 'mean'.

    Attributes:
        input (ndarray): The input tensor.
        target (ndarray): The target tensor.
        label_num (int): The number of labels.
        model (object): The model object.
        reduction (str): Specifies the reduction to apply to the output.
        softmax (ndarray): The softmax result, used for backward propagation.

    Methods:
        forward(input, target):
            Computes the forward pass of the cross-entropy loss.

        backward():
            Performs backward propagation by computing the input gradient and calling the model's backward method.
    """

    def __init__(self, model, label_num, reduction='mean'):
        self.input = None
        self.target = None
        self.label_num = label_num
        self.model = model
        self.reduction = reduction
        self.softmax = None  # Used to store the softmax result for backward propagation

    def forward(self, input, target):
        """
        Computes the forward pass of the cross-entropy loss.

        Args:
            input (ndarray): The input tensor.
            target (ndarray): The target tensor.

        Returns:
            float: The computed loss value.
        """
        self.input = input
        target = np.eye(self.label_num)[target]
        self.target = target

        input_exp = np.exp(input - np.max(input, axis=input.ndim - 1, keepdims=True))
        row_sums = np.sum(input_exp, axis=input.ndim - 1, keepdims=True)
        self.softmax = input_exp / row_sums
        losses = np.sum(-np.log(self.softmax) * target, axis=input.ndim - 1)
        if self.reduction == 'mean':
            loss = np.mean(losses)
        elif self.reduction == 'sum':
            loss = np.sum(losses)
        return loss  # Only return the loss value

    def backward(self):
        """
        Performs backward propagation by computing the input gradient and calling the model's backward method.
        """
        input_grad = self.softmax - self.target
        self.model.backward(input_grad)  # Backward propagate the input gradient

class MSELoss(Loss):

    def __init__(self, model) -> None:
        self.input = None
        self.target = None
        self.model = model

    def forward(self, input, target):
        self.input = input
        self.target = target
        loss = np.mean(np.square(input - target))
        
        return loss

    def backward(self):
        input_grad = 2 * (self.input - self.target) / len(self.target)

        self.model.backward(input_grad)

