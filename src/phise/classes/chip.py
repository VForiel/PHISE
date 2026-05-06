import numpy as np
import astropy.units as u
from typing import Optional, Tuple

from ..modules import laugiergram

class Chip:

    def __init__(self):
        ...

    def get_output_fields(self, ψ: np.ndarray[complex], λ: u.Quantity, φ: Optional[u.Quantity]=None, σ: Optional[u.Quantity]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        raise NotImplementedError # implemented in subclasses

    def laugiergram(self, show=False):
        import matplotlib.pyplot as plt

        ψs = np.eye(self.nb_inputs, dtype=np.complex128)

        phasors = np.empty((self.nb_inputs, self.nb_raw_outputs), dtype=np.complex128)
        for i in range(self.nb_inputs):
            phasors[i] = self.get_output_fields(ψs[i], self.parent_interferometer.λ)

        fig, axs = plt.subplots(1, self.nb_raw_outputs, figsize=(3 * self.nb_raw_outputs, 3), subplot_kw={'projection': 'polar'}, sharey=True, tight_layout=True)
        for i in range(self.nb_raw_outputs):
            laugiergram(phasors[:, i], ax=axs[i], labels=[f'Input {j+1}' for j in range(self.nb_inputs)] if i==0 else None, relative=True)
            axs[i].set_title(f'Output {i}')

        if show:
            plt.show()
        
        return fig