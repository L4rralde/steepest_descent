"""
Author: Emmanuel Larralde
"""
import numpy as np

from src.differentiable import C2class


class SteepestDescent:
    """
    Steepest Gradient Descent algorithm
    """
    FIXED_ALPHA = 0.0001

    def __init__(self, function: C2class, alpha_estimator: str = "") -> None:
        self.f = function
        self.prev_x = None
        self.x = None
        self.prev_alpha = SteepestDescent.FIXED_ALPHA
        self.grad = None

        match alpha_estimator:
            case "FIXED_STEP":
                self.estimate_step_size = self.fixed_step_size
            case "APPROX1_STEP":
                self.estimate_step_size = self.approx1_step_size
            case "APPROX2_STEP":
                self.estimate_step_size = self.approx2_step_size
            case _:
                self.estimate_step_size = self.fixed_step_size

    def fixed_step_size(self) -> float:
        """
        The simplest option to estimate best step size: a fixed one.
        """
        return SteepestDescent.FIXED_ALPHA

    def approx1_step_size(self) -> float:
        """
        Another option to estimate best step size.
        This one uses the gradient and the Hessian.
        """
        grad = self.f.gradient(self.x)
        hess = self.f.hessian(self.x)
        return (
            np.dot(grad, grad)/
            np.matmul(grad.T, np.matmul(hess, grad))
        )

    def approx2_step_size(self) -> float:
        """
        Another function to estimate best step size.
        This one uses past learning rate, evaluations and the gradient.
        """
        grad = self.f.gradient(self.x)
        f_x = self.f.eval(self.x)
        f_prevx = self.f.eval(self.prev_x)
        sq_norm_grad = np.dot(grad, grad)

        return (
            0.5*sq_norm_grad*self.prev_alpha**2/
            (f_prevx - f_x + self.prev_alpha*sq_norm_grad)
        )

    def met_stop_criteria(self, tf: float, tx: float) -> bool:
        """
        Checks if the algorithm has plateaued
        """
        f_x = self.f.eval(self.x)
        f_prevx = self.f.eval(self.prev_x)
        f_criterion = abs(f_x - f_prevx)/max(1, abs(f_x))
        if f_criterion <= tf:
            return True
        x_criterion = (
            np.linalg.norm(self.x - self.prev_x)/
            (max(1, np.linalg.norm(self.x)))
        )
        if x_criterion <= tx:
            return True
        return False

    def step(self) -> None:
        """
        Perfmors one cycle of the algorithm
        """
        self.grad = self.f.gradient(self.x)

        alpha = self.estimate_step_size()

        self.prev_x = np.copy(self.x)
        self.x += -alpha * self.grad
        self.prev_alpha = alpha

    def pre_solve(self, x0: np.array) -> None:
        """
        Performs the initial step of the algortihm.
        Required to avoid repeating alpha, and x
        """
        grad = self.f.gradient(x0)
        alpha = SteepestDescent.FIXED_ALPHA
        self.x = x0 - alpha*grad
        self.prev_x = np.copy(x0)
        self.prev_alpha = alpha

    def solve(
        self,
        x0: np.array,
        tf: float,
        tx: float,
        num_its: int = 1000000
    ) -> tuple:
        """
        Finds a local minima of a function given initial guess
        tf, tx thresholds and a maximum number of allowed cycles.
        """
        self.pre_solve(x0)
        k = 1
        for _ in range(num_its - 1):
            self.step()
            if self.met_stop_criteria(tf, tx):
                break
            k += 1

        return self.x, k

    def solve_step_by_step(
        self,
        x0: np.array,
        tf: float,
        tx: float,
        num_its: int = 1000000
    ) -> object:
        """
        Iterating version of self.solve()
        """
        self.pre_solve(x0)
        k = 1
        for _ in range(num_its - 1):
            self.step()
            if self.met_stop_criteria(tf, tx):
                break
            yield(
                self.x,
                k,
                np.linalg.norm(self.grad)
            )
            k += 1
