import numpy as np

class MMI:
    """Multi-Mode Interferometer (MMI).

    Args:
        matrix (np.ndarray): Unitary transfer matrix (U @ U* = I).
        λ0 (float): Reference wavelength.
        name (str): Readable name.
    """

    def __init__(self, matrix: np.ndarray, λ0: float, name: str):
        self.matrix = matrix
        self.λ0 = λ0
        self.name = name

    @property
    def matrix(self) -> np.ndarray:
        """Unitary transfer matrix.

        Returns:
            np.ndarray: Complex-valued square unitary matrix.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, value: np.ndarray):
        """Set unitary transfer matrix.

        Args:
            value (np.ndarray): Complex-valued square matrix.

        Raises:
            ValueError: If not convertible to complex array or not unitary.
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
        """Reference wavelength.

        Returns:
            float: Reference wavelength value.
        """
        return self._λ0

    @λ0.setter
    def λ0(self, value: float):
        """Set reference wavelength.

        Args:
            value (float): Reference wavelength.

        Raises:
            ValueError: If not a float-like value.
        """
        if not isinstance(value, (float, int)):
            raise ValueError('λ0 must be a float.')
        self._λ0 = float(value)

    @property
    def name(self) -> str:
        """MMI name.

        Returns:
            str: Readable name.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """Set MMI name.

        Args:
            value (str): Readable name.

        Raises:
            ValueError: If not a string.
        """
        if not isinstance(value, str):
            raise ValueError('Name must be a string.')
        self._name = value

    def propagate(self, input_signal: np.ndarray, λ: float) -> np.ndarray:
        """Propagate an input signal through the MMI.

        Args:
            input_signal (np.ndarray): Complex input vector.
            λ (float): Wavelength.

        Returns:
            np.ndarray: Complex output vector with same size as input.

        Raises:
            ValueError: If input size doesn't match the matrix input dimension.
        """
        input_signal = np.array(input_signal, dtype=complex)
        if input_signal.shape[0] != self.matrix.shape[0]:
            raise ValueError('Input signal size must match the number of input ports of the MMI.')
        M = self.matrix * np.exp(1j * (λ - self.λ0) / self.λ0)
        return M @ input_signal