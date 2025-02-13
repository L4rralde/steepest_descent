"""
2 times differentiable functions
Author: Emmanuel Larralde
"""

import numpy as np
from numba import njit


class C2class:
    """
    Abstract class of 2 times smooth differentiable function.
    These classes can be statical.
    """
    @staticmethod
    def eval(x: np.array) -> np.array:
        """Evaluates the function at x"""
        raise NotImplementedError

    @staticmethod
    def gradient(x: np.array) -> np.array:
        """Computes the gradient at x"""
        raise NotImplementedError

    @staticmethod
    def hessian(x: np.array) -> np.array:
        """Computes the Hessian at x"""
        raise NotImplementedError


class Rosenbrock:
    """
    Len agnostic Rosenbrock function.
    """
    @staticmethod
    @njit
    def eval(x: np.array) -> np.array:
        """
        Evaluates the rosenbrock at x.
        """
        return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    @staticmethod
    @njit
    def gradient(x: np.array) -> np.array:
        """
        Gradient of rosenborck at x.
        """
        n = len(x)
        result = np.zeros(n)
        result[:-1] = -400*x[:-1]*(x[1:] -  x[:-1]**2) - 2*(1 - x[:-1])
        result[1:] += 200*(x[1:] - x[:-1]**2)
        return result

    @staticmethod
    @njit
    def hessian(x: np.array) -> np.array:
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

class MdsCost:
    """
    Multidemensional Scaling cost function class
    """
    @staticmethod
    @njit
    def dist_mat(data: np.array, p: int) -> np.array:
        """
        Computes the distance matrix.
        """
        d = np.zeros((p, p))
        for i in range(p):
            for j in range(i):
                d[i][j] = np.linalg.norm(data[i] - data[j])
        return d + d.T

    @staticmethod
    def cost(delta: np.array, z: np.array, p: int) -> float:
        """
        Computes the cost for a given original space distance matrix
        and a matrix of the new space vectors.
        """
        acc = 0
        for i in range(p):
            for j in range(i):
                acc += (delta[i][j] - np.linalg.norm(z[i] - z[j]))**2
        return acc

    @staticmethod
    def dz_gradient(delta: np.array, z: np.array, p: int) -> np.array:
        """
        Computes the gradient of the evaluation of the cost function above.
        """
        d = MdsCost.dist_mat(z, p)
        g = np.zeros((p, len(z[0])))
        for k in range(p):
            for j in range(p):
                if d[k][j] == 0:
                    continue
                g[k] += (z[k] - z[j]) * (d[k][j] - delta[k][j])/d[k][j]
        return 2*g

    def __init__(self, data: np.array) -> None:
        """
        Initializes the class with the number of samples and
        the constant distance matrix (original space)
        """
        self.p, _ = data.shape
        self.delta = MdsCost.dist_mat(data, self.p)

    def eval(self, z: np.array) -> float:
        """
        Evaluates the cost function given the matrix of new arrays.
        """
        return MdsCost.cost(self.delta, z, self.p)

    def gradient(self, z: np.array) -> np.array:
        """
        Computes the gradient given the matrix of new arrays.
        """
        return MdsCost.dz_gradient(self.delta, z, self.p)

    def hessian(self, z: np.array) -> np.array:
        """
        Raises an error if tries to compute Hessian. It is not a matrix for this case.
        """
        raise NotImplementedError
