"""
Author: Emmanuel Larralde
"""

class SteepestDescent:
    def __init__(self, f: object) -> None:
        self.f = f
        self.k = 0

    