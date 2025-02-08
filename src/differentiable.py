"""
Author: Emmanuel Larralde
"""

import numpy as np


class C2class:
    def __call__(self, x: np.array) -> np.array:
        pass

    def gradient(self, x: np.array) -> np.array:
        pass

    def hessian(self, x: np.array) -> np.array:
        pass

class Rosenbrock:
    def __call__(self, x: np.array) -> np.array:
        return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def gradient(self, x: np.array) -> np.array:
        n = len(x)
        result = np.zeros(n)
        result[:-1] = -400*x[:-1]*(x[1:] -  x[:-1]**2) - 2*(1 - x[:-1])
        result[1:] += 200*(x[1:] - x[:-1]**2)
        return result

    def hessian(self, x: np.array) -> np.array:
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
