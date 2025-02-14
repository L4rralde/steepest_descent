"""Module for profiling optimization solvers using multiprocessing."""

from time import time
import multiprocessing

import numpy as np
import pandas as pd

from src.differentiable import Rosenbrock, MdsCost
from src.descent import SteepestDescent

class Profiler:
    """Base class for profiling optimization solvers."""

    def __init__(self, estimator="FIXED_STEP"):
        """Initializes the Profiler with the given estimator.

        Args:
            estimator (str): The step size estimation method for the solver.
        """
        self.estimator = estimator

    def run_solver(self, *shape):
        """Runs the solver for a single iteration. Must be implemented by subclasses.

        Args:
            shape (tuple): Shape of the input to the solver.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def profile(self, nruns: int, *shape):
        """Runs the solver multiple times in parallel and records execution times.

        Args:
            nruns (int): Number of runs to execute.
            shape (tuple): Shape of the input to the solver.

        Returns:
            list: Execution times for each run.
        """
        with multiprocessing.get_context("fork").Pool() as pool:
            records = pool.starmap(self.run_solver, [(*shape,)] * nruns)
        return records

class RosenbrockProfiler(Profiler):
    """Profiler for the Rosenbrock function optimization."""
    def run_solver(self, *shape):
        """Runs the Rosenbrock solver and records execution time.

        Args:
            shape (tuple): Shape of the input to the solver.

        Returns:
            float: Execution time of the solver.
        """
        solver = SteepestDescent(Rosenbrock, self.estimator)
        x0 = np.random.rand(*shape)
        start = time()
        solver.solve(x0, tf=1e-9, tx=1e-9)
        return time() - start

class MdsProfiler(Profiler):
    """Profiler for Multi-Dimensional Scaling (MDS) cost optimization."""
    def __init__(self, estimator="FIXED_STEP"):
        """Initializes the MDS profiler and loads sample data.

        Args:
            estimator (str): The step size estimation method for the solver.
        """
        super().__init__(estimator)
        df = pd.read_csv("data/iris.csv")
        self.samples = df.drop(columns=["variety"]).values

    def run_solver(self, *shape):
        """Runs the MDS solver and records execution time.

        Args:
            shape (tuple): Shape of the input to the solver (num_samples, dimensions).

        Returns:
            float: Execution time of the solver.
        """
        nsamples, dim = shape
        rand_idcs = np.random.choice(range(len(self.samples)), nsamples)
        sub_samples = self.samples[rand_idcs]
        solver = SteepestDescent(MdsCost(sub_samples), self.estimator)
        x0 = np.random.rand(nsamples, dim)
        start = time()
        solver.solve(x0, tf=1e-9, tx=1e-9)
        return time() - start
