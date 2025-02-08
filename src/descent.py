"""
Author: Emmanuel Larralde
"""
import numpy as np

from differentiable import C2class, Rosenbrock


class SteepestDescent:
    FIXED_ALPHA = 0.0001

    def __init__(self, function: C2class) -> None:
        self.f = function
        self.prev_x = None
        self.x = None
        self.prev_alpha = SteepestDescent.FIXED_ALPHA

    def fixed_step_size(self) -> float:
        return SteepestDescent.FIXED_ALPHA

    def approx1_step_size(self) -> float:
        grad = self.f.gradient(self.x)
        hess = self.f.hessian(self.x)
        return (
            np.dot(grad, grad)/
            np.matmul(grad.T, np.matmul(hess, grad))
        )

    def approx2_step_size(self) -> float:
        grad = self.f.gradient(self.x)
        f_x = self.f(self.x)
        f_prevx = self.f(self.prev_x)
        sq_norm_grad = np.dot(grad, grad)

        print(f_x, f_prevx)
        return (
            0.5*sq_norm_grad*self.prev_alpha**2/
            (f_prevx - f_x + self.prev_alpha*sq_norm_grad)
        )

    def met_stop_criteria(self, tf: float, tx: float) -> bool:
        f_x = self.f(self.x)
        f_prevx = self.f(self.prev_x)
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

    def solve(
        self,
        x0: np.array,
        tf: float,
        tx: float,
        num_its: int = 1000
    ) -> tuple:
        grad = self.f.gradient(x0)
        alpha = SteepestDescent.FIXED_ALPHA
        self.x = x0 - alpha*grad
        self.prev_x = np.copy(x0)
        self.prev_alpha = alpha
        k = 1

        for _ in range(num_its):
            grad = self.f.gradient(self.x)

            alpha = self.approx2_step_size()

            self.prev_x = np.copy(self.x)
            self.x += -alpha * grad
            self.prev_alpha = alpha

            if self.met_stop_criteria(tf, tx):
                break
            
            k += 1

        print(self.f(self.x))
        return self.x, k

def main() -> None:
    solver = SteepestDescent(Rosenbrock())
    x_star, k = solver.solve(
        np.zeros(2),
        1e-15,
        1e-15
    )


if __name__ == '__main__':
    main()
