"""Kernel nuller utilities and model.

Ce module contient la classe KernelNuller qui représente la partie
"nulling" d'un interféromètre à 4 télescopes. La classe fournit
les paramètres de configuration (phases appliquées, erreurs, ordre
des sorties, atténuations d'entrée) et des méthodes pour propager
les champs optiques et visualiser des grandeurs liées.
"""
import numpy as np
import numba as nb
import astropy.units as u
from typing import Tuple, Any, Optional
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from io import BytesIO
from LRFutils import color
from copy import deepcopy as copy
from ..modules import mmi
from ..modules import phase

class KernelNuller:
    """Représentation d'un kernel-nuller pour 4 télescopes.

    Attributs principaux
    ---------------------
    - φ: tableau des 14 phases/OPD injectées (Quantity)
    - σ: tableau des 14 erreurs internes d'OPD (Quantity)
    - λ0: longueur d'onde de référence (Quantity)
    - output_order: permutation décrivant l'ordre des sorties
    - input_attenuation: atténuations appliquées aux entrées (taille 4)
    - input_opd: OPD différentiel sur chaque entrée (taille 4, Quantity)
    
    Exemple
    -------
    >>> kn = KernelNuller(φ=..., σ=..., λ0=1*u.um)
    >>> nulls, darks, bright, total = kn.propagate_fields(ψ, 1*u.um)
    """
    __slots__ = ('_parent_interferometer', '_φ', '_σ', '_λ0', '_output_order', '_input_attenuation', '_input_opd', '_name')

    def __init__(self, φ: np.ndarray[u.Quantity], σ: np.ndarray[u.Quantity], λ0: u.Quantity, output_order: np.ndarray[int]=None, input_attenuation: np.ndarray[float]=None, input_opd: np.ndarray[u.Quantity]=None, name: str='Unnamed Kernel-Nuller'):
        """Initialise un KernelNuller.

        Paramètres
        ----------
        φ : astropy.units.Quantity
            Tableau (14,) des OPD appliqués (unités de longueur).
        σ : astropy.units.Quantity
            Tableau (14,) des erreurs intrinsèques d'OPD.
        λ0 : astropy.units.Quantity
            Longueur d'onde de référence à laquelle les matrices sont définies.
        output_order : array-like, optionnel
            Ordre des sorties (6 éléments) définissant les paires de sorties.
        input_attenuation : array-like de float, optionnel
            Atténuations appliquées aux 4 entrées optiques.
        input_opd : astropy.units.Quantity, optionnel
            OPD relatifs appliqués aux 4 entrées (taille 4).
        name : str, optionnel
            Nom descriptif de l'objet.
        """
        self._parent_interferometer = None
        self.φ = φ
        self.σ = σ
        self.λ0 = λ0
        self.output_order = output_order if output_order is not None else np.array([0, 1, 2, 3, 4, 5])
        self.input_attenuation = input_attenuation if input_attenuation is not None else np.array([1.0, 1.0, 1.0, 1.0])
        self.input_opd = input_opd if input_opd is not None else np.zeros(4) * u.m
        self.name = name

    def __str__(self) -> str:
        res = f'Kernel-Nuller "{self.name}"\n'
        res += f"  φ: [{', '.join([f'{i:.2e}' for i in self.φ.value])}] {self.φ.unit}\n"
        res += f"  σ: [{', '.join([f'{i:.2e}' for i in self.σ.value])}] {self.σ.unit}\n"
        res += f"  Output order: [{', '.join([f'{i}' for i in self.output_order])}]\n"
        res += f"  Input attenuation: [{', '.join([f'{i:.2e}' for i in self.input_attenuation])}]\n"
        res += f"  Input OPD: [{', '.join([f'{i:.2e}' for i in self.input_opd.value])}] {self.input_opd.unit}"
        return res.replace('e+00', '')

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def φ(self):
        """OPD/phase appliquée par élément du nuller.

        Retourne un astropy Quantity de forme (14,) exprimé en unités de
        longueur (m par exemple).
        """
        return self._φ

    @φ.setter
    def φ(self, φ: np.ndarray[u.Quantity]):
        """"φ.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if type(φ) != u.Quantity:
            raise ValueError('φ must be a Quantity')
        try:
            φ.to(u.m)
        except u.UnitConversionError:
            raise ValueError('φ must be in a distance unit')
        if φ.shape != (14,):
            raise ValueError('φ must have a shape of (14,)')
        if np.any(φ < 0):
            raise ValueError('φ must be positive')
        self._φ = φ

    @property
    def σ(self):
        """Erreurs intrinsèques d'OPD du nuller.

        Quantity shape (14,), même unité que `φ`.
        """
        return self._σ

    @σ.setter
    def σ(self, σ: np.ndarray[u.Quantity]):
        """"σ.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if type(σ) != u.Quantity:
            raise ValueError('σ must be a Quantity')
        try:
            σ.to(u.m)
        except u.UnitConversionError:
            raise ValueError('σ must be in a distance unit')
        if σ.shape != (14,):
            raise ValueError('σ must have a shape of (14,)')
        self._σ = σ

    @property
    def λ0(self):
        """Longueur d'onde de référence du modèle.

        Retourne un astropy Quantity (par ex. en m).
        """
        return self._λ0

    @λ0.setter
    def λ0(self, λ0: u.Quantity):
        """"λ0.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(λ0, u.Quantity):
            raise TypeError('λ0 must be an astropy Quantity')
        try:
            λ0 = λ0.to(u.m)
        except u.UnitConversionError:
            raise ValueError('λ0 must be in a distance unit')
        self._λ0 = λ0

    @property
    def output_order(self):
        """Ordre des sorties du nuller.

        Retourne un tableau d'entiers de taille 6 décrivant l'ordre des
        sorties et la structure des paires.
        """
        return self._output_order

    @output_order.setter
    def output_order(self, output_order: np.ndarray[int]):
        """"output_order.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        try:
            output_order = np.array(output_order, dtype=int)
        except:
            raise ValueError(f'output_order must be an array of integers, not {type(output_order)}')
        if output_order.shape != (6,):
            raise ValueError(f'output_order must have a shape of (6,), not {output_order.shape}')
        if not np.all(np.sort(output_order) == np.arange(6)):
            raise ValueError(f'output_order must contain all the integers from 0 to 5, not {output_order}')
        if output_order[0] - output_order[1] not in [-1, 1] or output_order[2] - output_order[3] not in [-1, 1] or output_order[4] - output_order[5] not in [-1, 1]:
            raise ValueError(f'output_order contain an invalid configuration of output pairs. Found {output_order}')
        self._output_order = output_order

    @property
    def input_attenuation(self):
        """Atténuations appliquées aux entrées.

        Tableau flottant de taille 4 (facteurs multiplicatifs).
        """
        return self._input_attenuation

    @input_attenuation.setter
    def input_attenuation(self, input_attenuation: np.ndarray[float]):
        """"input_attenuation.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        try:
            input_attenuation = np.array(input_attenuation, dtype=float)
        except:
            raise ValueError(f'input_attenuation must be an array of floats, not {type(input_attenuation)}')
        if input_attenuation.shape != (4,):
            raise ValueError(f'input_attenuation must have a shape of (4,), not {input_attenuation.shape}')
        self._input_attenuation = input_attenuation

    @property
    def input_opd(self):
        """OPD relatif appliqué sur chaque entrée.

        Quantity shape (4,) en unité de longueur.
        """
        return self._input_opd

    @input_opd.setter
    def input_opd(self, input_opd: np.ndarray[u.Quantity]):
        """"input_opd.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if type(input_opd) != u.Quantity:
            raise ValueError('input_opd must be a Quantity')
        try:
            input_opd.to(u.m)
        except u.UnitConversionError:
            raise ValueError('input_opd must be in a distance unit')
        if input_opd.shape != (4,):
            raise ValueError('input_opd must have a shape of (4,)')
        self._input_opd = input_opd

    @property
    def name(self):
        """Nom descriptif de l'instance."""
        return self._name

    @name.setter
    def name(self, name: str):
        """"name.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(name, str):
            raise ValueError('name must be a string')
        self._name = name

    @property
    def parent_interferometer(self):
        """Interféromètre parent associé à ce kernel-nuller.

        Propriété en lecture seule définie lors de l'association avec
        un objet Interferometer.
        """
        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, parent_interferometer):
        """"parent_interferometer.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        raise ValueError('parent_interferometer is read-only')

    def propagate_fields(self, ψ: np.ndarray[complex], λ: u.Quantity) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Propager numériquement les champs à travers le kernel-nuller.

        Cette méthode simule la propagation optique pour 4 entrées à une
        longueur d'onde donnée, en tenant compte des atténuations et des
        OPD d'entrée. Elle renvoie les champs électriques complexes des
        sorties null, dark et bright.

        Paramètres
        ----------
        ψ : ndarray de complex
            Champs électriques entrants pour les 4 voies (shape (4,)).
        λ : astropy.units.Quantity
            Longueur d'onde utilisée pour la propagation.

        Returns
        -------
        tuple
            (null_fields, dark_fields, bright_fields, total_bright)
            - null_fields : ndarray complex, shape (3,)
            - dark_fields : ndarray complex, shape (6,)
            - bright_fields : ndarray complex, shape (1,) ou scalar
            - total_bright : float
        """
        φ = self.φ.to(λ.unit).value
        σ = self.σ.to(λ.unit).value
        λ0 = self.λ0.to(λ.unit).value
        ψ *= self.input_attenuation
        ψ *= np.exp(-1j * 2 * np.pi * self.input_opd.to(λ.unit).value / λ.value)
        return propagate_fields_njit(ψ=ψ, φ=φ, σ=σ, λ=λ.value, λ0=λ0, output_order=self.output_order)

    def plot_phase(self, λ: u.Quantity, ψ: Optional[np.ndarray]=None, plot: bool = True) -> Optional[Any]:
        """Tracer les phases et amplitudes complexes en sortie du nuller.

        Cette méthode calcule la réponse des sorties pour chacun des 4
        signaux d'entrée isolés et trace la phase/amplitude des sorties
        null, dark et bright sur des diagrammes polaires.

        Paramètres
        ----------
        λ : astropy.units.Quantity ou convertible
            Longueur d'onde pour la simulation.
        ψ : array-like, optionnel
            Vecteur des amplitudes complexes d'entrée (par défaut [0.5,...]).
        plot : bool
            Si True, affiche la figure; si False, retourne les données.
        """
        if ψ is None:
            ψ = np.array([0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j])
        ψ1 = np.array([ψ[0], 0, 0, 0])
        ψ2 = np.array([0, ψ[1], 0, 0])
        ψ3 = np.array([0, 0, ψ[2], 0])
        ψ4 = np.array([0, 0, 0, ψ[3]])
        (n1, d1, b1) = self.propagate_fields(ψ1, λ)
        (n2, d2, b2) = self.propagate_fields(ψ2, λ)
        (n3, d3, b3) = self.propagate_fields(ψ3, λ)
        (n4, d4, b4) = self.propagate_fields(ψ4, λ)
        n2 = np.abs(n2) * np.exp(1j * (np.angle(n2) - np.angle(n1)))
        n3 = np.abs(n3) * np.exp(1j * (np.angle(n3) - np.angle(n1)))
        n4 = np.abs(n4) * np.exp(1j * (np.angle(n4) - np.angle(n1)))
        d2 = np.abs(d2) * np.exp(1j * (np.angle(d2) - np.angle(d1)))
        d3 = np.abs(d3) * np.exp(1j * (np.angle(d3) - np.angle(d1)))
        d4 = np.abs(d4) * np.exp(1j * (np.angle(d4) - np.angle(d1)))
        b2 = np.abs(b2) * np.exp(1j * (np.angle(b2) - np.angle(b1)))
        b3 = np.abs(b3) * np.exp(1j * (np.angle(b3) - np.angle(b1)))
        b4 = np.abs(b4) * np.exp(1j * (np.angle(b4) - np.angle(b1)))
        n1 = np.abs(n1) * np.exp(1j * 0)
        d1 = np.abs(d1) * np.exp(1j * 0)
        b1 = np.abs(b1) * np.exp(1j * 0)
        (_, axs) = plt.subplots(2, 6, figsize=(20, 7.5), subplot_kw={'projection': 'polar'})
        axs[0, 0].scatter(np.angle(b1), np.abs(b1), color='yellow', label='Input 1', alpha=0.5)
        axs[0, 0].plot([0, np.angle(b1)], [0, np.abs(b1)], color='yellow', alpha=0.5)
        axs[0, 0].scatter(np.angle(b2), np.abs(b2), color='green', label='Input 2', alpha=0.5)
        axs[0, 0].plot([0, np.angle(b2)], [0, np.abs(b2)], color='green', alpha=0.5)
        axs[0, 0].scatter(np.angle(b3), np.abs(b3), color='red', label='Input 3', alpha=0.5)
        axs[0, 0].plot([0, np.angle(b3)], [0, np.abs(b3)], color='red', alpha=0.5)
        axs[0, 0].scatter(np.angle(b4), np.abs(b4), color='blue', label='Input 4', alpha=0.5)
        axs[0, 0].plot([0, np.angle(b4)], [0, np.abs(b4)], color='blue', alpha=0.5)
        axs[0, 0].set_title('Bright output')
        for n in range(3):
            axs[0, n + 1].scatter(np.angle(n1[n]), np.abs(n1[n]), color='yellow', label='Input 1', alpha=0.5)
            axs[0, n + 1].plot([0, np.angle(n1[n])], [0, np.abs(n1[n])], color='yellow', alpha=0.5)
            axs[0, n + 1].scatter(np.angle(n2[n]), np.abs(n2[n]), color='green', label='Input 2', alpha=0.5)
            axs[0, n + 1].plot([0, np.angle(n2[n])], [0, np.abs(n2[n])], color='green', alpha=0.5)
            axs[0, n + 1].scatter(np.angle(n3[n]), np.abs(n3[n]), color='red', label='Input 3', alpha=0.5)
            axs[0, n + 1].plot([0, np.angle(n3[n])], [0, np.abs(n3[n])], color='red', alpha=0.5)
            axs[0, n + 1].scatter(np.angle(n4[n]), np.abs(n4[n]), color='blue', label='Input 4', alpha=0.5)
            axs[0, n + 1].plot([0, np.angle(n4[n])], [0, np.abs(n4[n])], color='blue', alpha=0.5)
            axs[0, n + 1].set_title(f'Null output {n + 1}')
        for d in range(6):
            axs[1, d].scatter(np.angle(d1[d]), np.abs(d1[d]), color='yellow', label='I1', alpha=0.5)
            axs[1, d].plot([0, np.angle(d1[d])], [0, np.abs(d1[d])], color='yellow', alpha=0.5)
            axs[1, d].scatter(np.angle(d2[d]), np.abs(d2[d]), color='green', label='I2', alpha=0.5)
            axs[1, d].plot([0, np.angle(d2[d])], [0, np.abs(d2[d])], color='green', alpha=0.5)
            axs[1, d].scatter(np.angle(d3[d]), np.abs(d3[d]), color='red', label='I3', alpha=0.5)
            axs[1, d].plot([0, np.angle(d3[d])], [0, np.abs(d3[d])], color='red', alpha=0.5)
            axs[1, d].scatter(np.angle(d4[d]), np.abs(d4[d]), color='blue', label='I4', alpha=0.5)
            axs[1, d].plot([0, np.angle(d4[d])], [0, np.abs(d4[d])], color='blue', alpha=0.5)
            axs[1, d].set_title(f'Dark output {d + 1}')
        m = np.max(np.concatenate([np.abs(n1), np.abs(n2), np.abs(n3), np.abs(n4), np.abs(d1), np.abs(d2), np.abs(d3), np.abs(d4), np.array([np.abs(b1), np.abs(b2), np.abs(b3), np.abs(b4)])]))
        for ax in axs.flatten():
            ax.set_ylim(0, m)
        axs[0, 4].axis('off')
        axs[0, 5].axis('off')
        axs[0, 0].legend()
        if not plot:
            plot = BytesIO()
            plt.savefig(plot, format='png')
            plt.close()
            return plot.getvalue()
        plt.show()

    def rebind_outputs(self, λ):
        """
        Correct the output order of the KernelNuller object. To do so, we successively obstruct two inputs and add a π/4 phase over one of the two remaining inputs. Doing so, 

        Parameters
        ----------
        - self: KernelNuller object
        - λ: Wavelength of the observation

        Returns
        -------
        - KernelNuller object
        """
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[3] = (1 + 0j) * np.sqrt(1 / 2)
        (_, d, _) = self.propagate_fields(ψ=ψ, λ=λ)
        k1 = np.argsort((d * np.conj(d)).real)[:2]
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[2] = (1 + 0j) * np.sqrt(1 / 2)
        (_, d, _) = self.propagate_fields(ψ=ψ, λ=λ)
        k2 = np.argsort((d * np.conj(d)).real)[:2]
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1 + 0j) * np.sqrt(1 / 2)
        (_, d, _) = self.propagate_fields(ψ=ψ, λ=λ)
        k3 = np.argsort((d * np.conj(d)).real)[:2]
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1 + 0j) * np.sqrt(1 / 2)
        ψ[1] *= np.exp(-1j * np.pi / 2)
        (_, d, _) = self.propagate_fields(ψ=ψ, λ=λ)
        dk1 = d[k1]
        diff = np.abs(dk1[0] - dk1[1])
        if diff < 0:
            k1 = np.flip(k1)
        dk2 = d[k2]
        diff = np.abs(dk2[0] - dk2[1])
        if diff < 0:
            k2 = np.flip(k2)
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1 + 0j) * np.sqrt(1 / 2)
        ψ[2] *= np.exp(-1j * np.pi / 2)
        (_, d, _) = self.propagate_fields(ψ=ψ, λ=λ)
        dk3 = d[k3]
        diff = np.abs(dk3[0] - dk3[1])
        if diff < 0:
            k3 = np.flip(k3)
        self.output_order = np.concatenate([k1, k2, k3])

@nb.njit()
def propagate_fields_njit(ψ: np.ndarray[complex], φ: np.ndarray[float], σ: np.ndarray[float], λ: float, λ0: float, output_order: np.ndarray[int]) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
    """
    Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach.
    ⚠️ Does not take in account the input attenuation and OPD.
    
    Parameters
    ----------
    - ψ: Array of 4 input signals complex amplitudes
    - φ: Array of 14 injected OPD (in wavelenght unit)
    - σ: Array of 14 intrasic OPD error (in wavelenght unit)
    - λ: Wavelength of the light
    - λ0: Reference wavelenght (in wavelenght unit)
    - output_order: Order of the outputs

    Returns
    -------
    - Array of 3 null outputs electric fields
    - Array of 6 dark outputs electric fields
    - Bright output electric fields
    """
    λ_ratio = λ0 / λ
    N = 1 / np.sqrt(2) * np.array([[1 + 0j, 1 + 0j], [1 + 0j, np.exp(-1j * np.pi * λ_ratio)]], dtype=np.complex128)
    Na = np.abs(N)
    Nφ = np.angle(N)
    N = Na * np.exp(1j * Nφ * λ_ratio)
    θ: float = np.pi / 2
    R = 1 / np.sqrt(2) * np.array([[np.exp(1j * θ / 2), np.exp(-1j * θ / 2)], [np.exp(-1j * θ / 2), np.exp(1j * θ / 2)]])
    Ra = np.abs(R)
    Rφ = np.angle(R)
    R = Ra * np.exp(1j * Rφ * λ_ratio)
    φ = phase.bound_njit(φ + σ, λ)
    nuller_inputs = phase.shift_njit(ψ, φ[:4], λ)
    N1 = np.dot(N, nuller_inputs[:2])
    N2 = N @ nuller_inputs[2:]
    N1_shifted = phase.shift_njit(N1, φ[4:6], λ)
    N2_shifted = phase.shift_njit(N2, φ[6:8], λ)
    N3 = N @ np.array([N1_shifted[0], N2_shifted[0]])
    N4 = N @ np.array([N1_shifted[1], N2_shifted[1]])
    nulls = np.array([N3[1], N4[0], N4[1]], dtype=np.complex128)
    bright = N3[0]
    R_inputs = np.array([N3[1], N3[1], N4[0], N4[0], N4[1], N4[1]]) * 1 / np.sqrt(2)
    R_inputs = phase.shift_njit(R_inputs, φ[8:], λ)
    R1_output = R @ np.array([R_inputs[0], R_inputs[2]])
    R2_output = R @ np.array([R_inputs[1], R_inputs[4]])
    R3_output = R @ np.array([R_inputs[3], R_inputs[5]])
    darks = np.array([R1_output[0], R1_output[1], R2_output[0], R2_output[1], R3_output[0], R3_output[1]], dtype=np.complex128)
    darks = darks[output_order]
    return (nulls, darks, bright)