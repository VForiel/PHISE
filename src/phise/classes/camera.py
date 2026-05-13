from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import astropy.units as u
import math
import numba as nb
import numpy as np

if TYPE_CHECKING:
    from .interferometer import Interferometer

#==============================================================================
# Numba accelerated functions
#==============================================================================

@nb.njit(cache=True)
def _get_flux_jit(
    psi: np.ndarray,
    e: float = 1.0,
    ideal: bool = False,
    qe: float = 1.0,
    dc: float = 100.0,
    ron: float = 10.0,
    fwc: float = math.inf,
    gain: float = 1.0,
    resolution: int = 1,
    spot_size: float = 1.0,
    dark:np.ndarray = None,
) -> int:

    return np.sum(_get_image_jit(
        psi=psi,
        e=e,
        ideal=ideal,
        qe=qe,
        dc=dc,
        ron=ron,
        fwc=fwc,
        gain=gain,
        resolution=resolution,
        spot_size=spot_size,
        dark=dark,
    ))

@nb.njit(cache=True)
def _get_image_jit(
    psi: np.ndarray,
    e: float = 1.0,
    ideal: bool = False,
    qe: float = 1.0,
    dc: float = 100.0,
    ron: float = 10.0,
    fwc: float = math.inf,
    gain: float = 1.0,
    resolution: int = 1,
    spot_size: float = 1.0,
    dark:np.ndarray = None,
) -> np.ndarray:
    """Numba-accelerated scalar detector model used for the 1-pixel fast path."""

    # Getting theoretical beam spot size
    flux = np.sum(np.abs(psi) ** 2) * e
    image = _get_spot_profile(resolution, spot_size, flux)

    if not ideal:

        # Apply quantum efficiency
        image *= qe

        for i in range(resolution):
            for j in range(resolution):

                # Photon noise
                if image[i,j] <= 2000000000.0:
                    image[i,j] = np.random.poisson(image[i,j])
                else: # Avoid overflow with Poisson method at high flux
                    image[i, j] = int(image[i,j] + np.random.normal(0.0, np.sqrt(image[i, j])))

                # Dark current noise
                dark_current = np.random.poisson(dc * e)

                # Read out noise
                readout_noise = np.random.normal(0.0, ron)

                # Total photo-electrons
                image[i,j] = image[i,j] + readout_noise + dark_current

                if image[i, j] > fwc:
                    image[i, j] = fwc

                image[i,j] = np.rint(image[i,j] / gain)

                if dark is not None:
                    image[i,j] = image[i,j] - dark[i, j]

    return image

@nb.njit(cache=True)
def _flux_to_amplitude(flux: float, sigma_x: float, sigma_y: float) -> float:
    """Convert a total Gaussian flux into a peak amplitude.

    The integrated flux of a 2D Gaussian is

    .. math::

        F = 2\\pi \\sigma_x \\sigma_y A

    where ``A`` is the peak amplitude.
    """

    return flux / (2.0 * math.pi * sigma_x * sigma_y)

@nb.njit(cache=True)
def _get_spot_profile(
    resolution: int,
    spot_size: float,
    flux: float,
) -> np.ndarray:
    """Draw a 2D Gaussian spot profile on a square grid."""

    x = np.zeros((resolution, resolution))
    y = np.zeros((resolution, resolution))
    for i in range(resolution):
        x[i, :] = np.arange(resolution) - (resolution - 1) / 2 
        y[:, i] = np.arange(resolution) - (resolution - 1) / 2

    σ = spot_size
    a = _flux_to_amplitude(flux, σ, σ)

    return a * np.exp(-0.5 * ((x / σ) ** 2 + (y / σ) ** 2))

#==============================================================================
# Camera class
#==============================================================================

class Camera:
    """Virtual camera used to simulate photon detection.

    Args:
        e (u.Quantity): Exposure time as an Astropy quantity in a time unit
            (for example ``1 * u.s``).
        ideal (bool): If ``True``, the camera is noise-free.
        name (str): Human-readable name for the camera.
        ron (float): Read-out noise in electrons rms. Default: 0.0.
        dc (float): Dark current in electrons/pixel/s. Default: 0.0.
        fwc (float): Full-well capacity in electrons. Default: infinity.
        qe (float): Quantum efficiency in the range [0, 1]. Default: 1.0.
        gain (float): Gain in electrons/ADU. Default: 1.0.
        resolution (int): Detector side length in pixels. Default: 1.
        spot_size (float): Standard deviation of the 2D Gaussian profile in pixels. Default: 1.0.
        hdr (list[float]): List of exposure times for HDR acquisition, relative to the primary exposure time. Default: None (no HDR).

    Raises:
        TypeError: If a typed argument has an invalid type.
        ValueError: If a physical or numeric parameter is out of bounds.
    """

    __slots__ = (
        '_parent_interferometer',
        '_e',
        '_e_unit',
        '_name',
        '_ideal',
        '_ron',
        '_dc',
        '_fwc',
        '_qe',
        '_gain',
        '_resolution',
        '_spot_size',
        '_hdr',
        '_dark',
        '_hdr_darks',
        '_update_dark_on_change',
    )

    def __init__(
        self,
        e: u.Quantity = 1 * u.s,
        ideal: bool = False,
        name: str = 'Unnamed Camera',
        ron: float = 10.0,
        dc: float = 100.0,
        fwc: float = math.inf,
        qe: float = 1.0,
        gain: float = 1.0,
        resolution: int = 1,
        spot_size: float = 1.0,
        hdr: list[float] = None,
    ):
        
        
        self._update_dark_on_change = False

        self._parent_interferometer = None
        self.ideal = ideal
        self.name = name
        self.ron = ron
        self.dc = dc
        self.fwc = fwc
        self.qe = qe
        self.gain = gain
        self.resolution = resolution
        self.spot_size = spot_size
        self.e = e
        self.hdr = hdr if hdr is not None else []

        self._dark = np.zeros((self._resolution, self._resolution))
        self._hdr_darks = [np.zeros((self._resolution, self._resolution)) for _ in range(len(self._hdr))]
        self.update_all_darks()
        self._update_dark_on_change = True

    # Special methods ---------------------------------------------------------

    def __str__(self) -> str:
        res = f'Camera "{self.name}"\n'
        res += f'  Exposure time: {self.e:.2g}\n'
        res += f'  QE: {(self.qe * 100):.2g}% | RON: {self.ron:.2g} e- | DC: {self.dc:.2g} e-/s | Gain: {self.gain:.2g} e-/ADU\n'
        res += f'  Resolution: {self.resolution} px | Spot size: {self.spot_size} px | HDR: {self.hdr} * exposure time\n'
        return res

    def __repr__(self) -> str:
        return self.__str__()

    # Properties --------------------------------------------------------------

    @property
    def e(self) -> u.Quantity:
        """Camera exposure time."""
        return (self._e * u.s).to(self._e_unit)

    @e.setter
    def e(self, e: u.Quantity):
        """Set the exposure time."""

        if not isinstance(e, u.Quantity):
            raise TypeError('e must be an astropy Quantity')
        try:
            e_val = e.to(u.s).value
        except u.UnitConversionError:
            raise ValueError('e must be in a time unit')
        if e_val <= 0:
            raise ValueError('e must be positive')

        # Check if the camera if not totally saturated by the dark current. Show a warning if the requested exposure time is above the dark saturation time (fwc/dc), which means that the dark current will saturate the camera even in the absence of any light.
        max_e = self.fwc / self.dc
        if e_val > max_e:
            print(f"⚠️ Warning: The requested exposure time {e:.2g} is above the dark saturation time {max_e:.2g}. The camera will be saturated by the dark current even in the absence of any light.")

        self._e_unit = e.unit
        self._e = e_val
        if self._update_dark_on_change:
            self.update_all_darks()

    @property
    def parent_interferometer(self) -> Interferometer:
        """Read-only reference to the parent interferometer."""

        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, _):
        raise ValueError('parent_interferometer is read-only')

    @property
    def ideal(self) -> bool:
        """Whether the camera is in ideal mode."""

        return self._ideal

    @ideal.setter
    def ideal(self, ideal: bool):
        if not isinstance(ideal, bool):
            raise TypeError('ideal must be a boolean')
        self._ideal = ideal

    @property
    def name(self) -> str:
        """Human-readable camera name."""

        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name

    @property
    def ron(self) -> float:
        """Read-out noise in electrons root mean square."""

        return self._ron

    @ron.setter
    def ron(self, val: float):
        self._ron = float(val)
        if self._update_dark_on_change:
            self.update_all_darks()

    @property
    def dc(self) -> float:
        """Dark current in electrons/pixel/second."""
        return self._dc

    @dc.setter
    def dc(self, val: float):
        self._dc = float(val)
        if self._update_dark_on_change:
            self.update_all_darks()

    @property
    def fwc(self) -> float:
        """Full-well capacity in electrons."""
        return self._fwc

    @fwc.setter
    def fwc(self, val: float):
        self._fwc = float(val)
        if self._update_dark_on_change:
            self.update_all_darks()

    @property
    def qe(self) -> float:
        """Quantum efficiency in the range [0, 1]."""
        return self._qe

    @qe.setter
    def qe(self, val: float):
        self._qe = float(val)
        if self._update_dark_on_change:
            self.update_all_darks()

    @property
    def gain(self) -> float:
        """Gain in electrons/ADU."""
        return self._gain

    @gain.setter
    def gain(self, val: float):
        self._gain = float(val)
        if self._update_dark_on_change:
            self.update_all_darks()

    @property
    def resolution(self) -> int:
        """Detector side length in pixels."""
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: int):
        if not isinstance(resolution, Integral) or isinstance(resolution, bool):
            raise TypeError('resolution must be an integer')
        resolution = int(resolution)
        if resolution < 1:
            raise ValueError('resolution must be positive')
        self._resolution = resolution
        if self._update_dark_on_change:
            self.update_all_darks()

    @property
    def spot_size(self) -> float:
        """Standard deviation of the 2D Gaussian spot profile in pixels."""
        return self._spot_size

    @spot_size.setter
    def spot_size(self, spot_size: float):
        if not isinstance(spot_size, (int, float)):
            raise TypeError('spot_size must be a number')
        if spot_size <= 0:
            raise ValueError('spot_size must be positive')
        self._spot_size = float(spot_size)
        if self._update_dark_on_change:
            self.update_all_darks() 

    @property
    def hdr(self) -> bool:
        """List of exposure times for HDR acquisition, relative to the primary exposure time."""
        return self._hdr

    @hdr.setter
    def hdr(self, hdr: list[float]):
        if not isinstance(hdr, list):
            raise TypeError('hdr must be a list of floats')
        for val in hdr:
            if not isinstance(val, (int, float)):
                raise TypeError('hdr must be a list of floats')
            if val <= 0:
                raise ValueError('hdr values must be positive')
        self._hdr = [float(val) for val in hdr]
        if self._update_dark_on_change:
            self.update_all_darks()

    # Public methods ----------------------------------------------------------

    def take_dark(self, N:int = 1000, e: u.Quantity = None) -> np.ndarray:
        """Estimate the master dark frame at a given exposure time.

        Dark acquisition must bypass current dark subtraction, otherwise the
        estimate becomes self-referential (new_dark ~= true_dark - old_dark).
        """

        if e is None:
            e = self.e
        if not isinstance(e, u.Quantity):
            raise TypeError('e must be an astropy Quantity')

        e_seconds = e.to(u.s).value
        darks = np.empty((N, self.resolution, self.resolution))
        for i in range(N):
            darks[i] = _get_image_jit(
                psi=np.array([0j], dtype=np.complex128),
                e=e_seconds,
                ideal=self._ideal,
                qe=self._qe,
                dc=self._dc,
                ron=self._ron,
                fwc=self._fwc,
                gain=self._gain,
                resolution=self._resolution,
                spot_size=self._spot_size,
                dark=None,
            )

        return np.mean(darks, axis=0)

    def update_all_darks(self):
        self._dark = self.take_dark(N=1000, e=self.e)
        self._hdr_darks = [self.take_dark(N=1000, e=self.e * hdr_factor) for hdr_factor in self.hdr]

    def get_flux(self, psi: np.ndarray[complex]) -> float:
        """Calculate the total flux at the camera plane given an input complex field.

        Args:
            psi (np.ndarray[complex]): 2D array representing the complex field at the camera plane.

        Returns:
            float: Total flux at the camera plane.
        """

        return _get_flux_jit(
            psi=np.array(psi, dtype=np.complex128),
            e=self._e,
            ideal=self._ideal,
            qe=self._qe,
            dc=self._dc,
            ron=self._ron,
            fwc=self._fwc,
            gain=self._gain,
            resolution=self._resolution,
            spot_size=self._spot_size,
            dark=self._dark,
        )

    def get_image(self, psi: np.ndarray[complex]) -> int:
        """Simulate the acquisition of an image given an input complex field.

        Args:
            psi (np.ndarray[complex]): 2D array representing the complex field at the camera plane.

        Returns:
            int | np.ndarray[int]: Output of the camera (either image if mode='raw' or aggregated values for other modes).
        """

        return _get_image_jit(
            psi=np.array(psi, dtype=np.complex128),
            e=self._e,
            ideal=self._ideal,
            qe=self._qe,
            dc=self._dc,
            ron=self._ron,
            fwc=self._fwc,
            gain=self._gain,
            resolution=self._resolution,
            spot_size=self._spot_size,
            dark=self._dark,
        )