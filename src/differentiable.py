"""
2 times differentiable functions
Author: Emmanuel Larralde
"""

import numpy as np


class C2class:
    """
    Abstract class of 2 times smooth differentiable function.
    These classes can be statical.
    """
    def __call__(self, x: np.array) -> np.array:
        """Evaluates the function at x"""
        raise NotImplementedError

    def gradient(self, x: np.array) -> np.array:
        """Computes the gradient at x"""
        raise NotImplementedError

    def hessian(self, x: np.array) -> np.array:
        """Computes the Hessian at x"""
        raise NotImplementedError

class Rosenbrock:
    """
    Len agnostick Rosenbrock function.
    """
    def __call__(self, x: np.array) -> np.array:
        """
        Evaluates the rosenbrock at x.
        """
        return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def gradient(self, x: np.array) -> np.array:
        """
        Gradient of rosenborck at x.
        """
        n = len(x)
        result = np.zeros(n)
        result[:-1] = -400*x[:-1]*(x[1:] -  x[:-1]**2) - 2*(1 - x[:-1])
        result[1:] += 200*(x[1:] - x[:-1]**2)
        return result

    def hessian(self, x: np.array) -> np.array:
        """
        Hessian of the rosenbrock function at x
        """
        n = len(x)
        sub_diagonal = -400*x[:-1]
        diagonal = np.zeros(n)
        diagonal[:-1] += -400*x[1:] + 1200*x[:-1]**2 + 2
        diagonal[1:] += 200
        return(
            np.diagflat(sub_diagonal, -1) +
            np.diagflat(diagonal) +
            np.diagflat(sub_diagonal, 1)
        )
