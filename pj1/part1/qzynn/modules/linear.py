import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Module

class Linear(Module):
    """
    A linear transformation module.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Whether to include bias term. Default is True.

    Attributes:
        input: Input tensor.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        params (dict): Dictionary of learnable parameters.
        grads (dict): Dictionary of gradients of parameters.

    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the Linear module.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Whether to include bias term. Default is True.

        """

        # input and output
        self.input = None
        self.in_features = in_features
        self.out_features = out_features

        # params
        self.params = {}
        k = 1 / in_features
        self.params['W'] = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(out_features, in_features))
        self.params['b'] = None
        if bias:
            self.params['b'] = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(out_features))

        # grads of params
        self.grads = {}

    def forward(self, input):
        """
        Performs forward pass of the Linear module.

        Args:
            input: Input tensor.

        Returns:
            output: Output tensor.

        """

        self.input = input

        output = np.dot(input, self.params['W'].T)
        if self.params['b'] is not None:
            output = output + self.params['b']
        return output

    def backward(self, output_grad):
        """
        Performs backward pass of the Linear module.

        Args:
            output_grad: Gradient of the output tensor.

        Returns:
            input_grad: Gradient of the input tensor.

        """

        self.grads['W'] = np.tensordot(output_grad.T, self.input, axes=(list(range(1, output_grad.ndim)), list(range(self.input.ndim - 2, -1, -1))))
        self.grads['b'] = np.sum(output_grad, axis=tuple(range(0, output_grad.ndim - 1)))
        input_grad = np.dot(output_grad, self.params['W'])

        assert self.grads['W'].shape == self.params['W'].shape
        assert self.grads['b'].shape == self.params['b'].shape
        assert input_grad.shape == self.input.shape

        return input_grad
