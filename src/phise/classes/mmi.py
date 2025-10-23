"""Module generated docstring."""
import numpy as np

class MMI:
    """"MMI class.

Attributes
----------
(Automatically added placeholder.)
"""

    def __init__(self, matrix: np.ndarray, λ0: float, name: str):
        self.matrix = matrix
        self.λ0 = λ0
        self.name = name

    @property
    def matrix(self) -> np.ndarray:
        """"matrix.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._matrix

    @matrix.setter
    def matrix(self, value: np.ndarray):
        """"matrix.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        try:
            value = np.array(value, dtype=complex)
        except Exception as e:
            raise ValueError('Matrix must be convertible to a numpy array of complex numbers.') from e
        if not np.allclose(value @ value.conj(), np.eye(value.shape[0])):
            raise ValueError('Matrix must be unitary (UU* = I).')
        self._matrix = value

    @property
    def λ0(self) -> float:
        """"λ0.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._λ0

    @λ0.setter
    def λ0(self, value: float):
        """"λ0.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(value, (float, int)):
            raise ValueError('λ0 must be a float.')
        self._λ0 = float(value)

    @property
    def name(self) -> str:
        """"name.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._name

    @name.setter
    def name(self, value: str):
        """"name.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(value, str):
            raise ValueError('Name must be a string.')
        self._name = value

    def propagate(self, input_signal: np.ndarray, λ: float) -> np.ndarray:
        """"propagate.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        input_signal = np.array(input_signal, dtype=complex)
        if input_signal.shape[0] != self.matrix.shape[0]:
            raise ValueError('Input signal size must match the number of input ports of the MMI.')
        M = self.matrix * np.exp(1j * (λ - self.λ0) / self.λ0)
        return M @ input_signal