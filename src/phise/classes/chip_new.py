

###############################################################################
# ⚠️ WORK IN PROGRESS ⚠️
# This file as well as the entire .subcomponents module will replace
# the current kernel_nuller module.
###############################################################################

from .subcomponents import Shifter
from .subcomponents import MMI
from .subcomponents import YSplitter

from astropy import units as u
import numpy as np

class Chip:

    def __init__(self, λ0=1550 * u.nm, layers=None):
        ...

    def add_layer(self, layer):
        ...

    def get_output_fields(self, input_fields, λ):
        ...

    def get_output_intensities(self, input_fields, λ):
        ...

    def get_kernels(self, input_fields, λ):
        ...

    def compute_intensities(self, output_fields):
        ...

    def compute_kernels(self, output_intensities):
        ...

    def compile(self):
        compiled_layers = [np.eye(len(self.layers[0].inputs), dtype=complex)]
        buffer = [np.eye(compiled_layers.shape[1], dtype=complex)]
        for layer in self.layers:
            if layer.is_static():
                buffer = np.array(layer.matrix, dtype=complex) @ buffer
            else:
                compiled_layers.append(buffer)
                if buffer != []:
                    buffer = []


    @staticmethod
    def get_superKN(σ_rms = 10 * u.nm, λ0=1550 * u.nm) -> 'Chip':
        """Get a predefined Super-KN Chip instance.

        Returns:
            Chip: Predefined Super-KN Chip.
        """
        
        chip = Chip()

        def get_random_opd(σ_rms, size):
            return np.abs(np.random.normal(0, σ_rms.to(u.nm).value, size) * u.nm)

        # 2x2 Nuller MMI matrix
        N = 1/np.sqrt(2) * [
            [1,  1],
            [1, -1]
        ]

        # 2x2 Recombiners MMI matrix
        θ = np.pi / 4
        R = 1/np.sqrt(2) * [
            [np.exp(1j * θ), np.exp(- 1j * θ)],
            [np.exp(- 1j * θ), np.exp(1j * θ)]
        ]

        # Perturbations
        chip.add_layer(Shifter(inputs=[0,1,2,3], static=True, default=get_random_opd(σ_rms, 4)))
        # First layer of shifters
        chip.add_layer(Shifter(inputs=[0,1,2,3]))
        # First layer of Nuller MMIs
        chip.add_layer(MMI(matrix=N, inputs=[0,1]))
        chip.add_layer(MMI(matrix=N, inputs=[2,3]))
        # Perturbations
        chip.add_layer(Shifter(inputs=[0,1,2,3], static=True, default=get_random_opd(σ_rms, 4)))
        # Second layer of shifters
        chip.add_layer(Shifter(inputs=[0,1,2,3]))
        # Second layer of Nuller MMIs
        chip.add_layer(MMI(matrix=N, inputs=[0,2]))
        chip.add_layer(MMI(matrix=N, inputs=[1,3]))
        # Spliter
        chip.add_layer(YSplitter(inputs=[1,2,3]))
        # Perturbations
        chip.add_layer(Shifter(inputs=[1,2,3,4,5,6], static=True, default=get_random_opd(σ_rms, 6)))
        # Final layer of shifters
        chip.add_layer(Shifter(inputs=[1,2,3,4,5,6]))
        # Cross Recombiners
        chip.add_layer(MMI(matrix=R, inputs=[1,3]))
        chip.add_layer(MMI(matrix=R, inputs=[2,5]))
        chip.add_layer(MMI(matrix=R, inputs=[4,6]))

        chip.compile()

        chip.kernel_combinations = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]

    