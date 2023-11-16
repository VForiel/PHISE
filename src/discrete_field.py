import numpy as np

class DiscretField:
    def __init__(self, value:np.ndarray):
        self.complex = value

    @property
    def intensity(self):
        return np.abs(self.complex)