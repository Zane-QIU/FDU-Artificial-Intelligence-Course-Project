
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from .base import Module

class Sigmoid(Module):
    """Applies the element-wise function:
    .. math::
    \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}

    Shape:
    - input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - output: :math:`(*)`, same shape as the input.
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.params = None

    def forward(self, input):

        output = 1 / (1 + np.exp(-input))

        self.output = output
        return output

    def backward(self, output_grad):
        """
        Input:
            - output_grad:(*)
            partial (loss function) / partial (output of this module)

        Return:
            - input_grad:(*)
            partial (loss function) / partial (input of this module)
        """

        input_grad = output_grad * ((1 - self.output) * self.output)

        return input_grad
    

class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise:
    ReLU(x) = max(0, x)

    Shape:
    - Input: (*), where * means any number of dimensions.
    - Output: (*), same shape as the input.
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.params = None

    def forward(self, input):
        self.input = input
        output = np.where(input > 0, input, 0)
        return output
    
    def backward(self, output_grad):
        input_grad = np.where(self.input > 0, output_grad, 0)
        return input_grad
    
class Tanh(Module):
    """
    Applies the hyperbolic tangent function element-wise:
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Shape:
    - Input: (*), where * means any number of dimensions.
    - Output: (*), same shape as the input.
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.params = None

    def forward(self, input):
        self.input = input
        output = np.tanh(input)
        return output
    
    def backward(self, output_grad):
        input_grad = output_grad * (1 - np.tanh(self.input) ** 2)
        return input_grad