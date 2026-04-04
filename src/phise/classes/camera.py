from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import astropy.units as u
import math
import numba as nb
import numpy as np

if TYPE_CHECKING:
    from .interferometer import Interferometer


def _flux_to_amplitude(flux: float, sigma_x: float, sigma_y: float) -> float:
    """Convert a total Gaussian flux into a peak amplitude.

    The integrated flux of a 2D Gaussian is

    .. math::

        F = 2\\pi \\sigma_x \\sigma_y A

    where ``A`` is the peak amplitude.
    """

    return flux / (2.0 * math.pi * sigma_x * sigma_y)


def _gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    amplitude: float,
    theta: float = 0.0,
) -> np.ndarray:
    """Evaluate a rotated 2D Gaussian on a pixel grid."""

    dx = x - x0
    dy = y - y0
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x_rot = dx * cos_theta + dy * sin_theta
    y_rot = -dx * sin_theta + dy * cos_theta
    exponent = -((x_rot**2) / (2.0 * sigma_x**2) + (y_rot**2) / (2.0 * sigma_y**2))
    return amplitude * np.exp(exponent)


def _estimate_flux_from_clipped_image(
    image: np.ndarray,
    dark_level: float,
    saturation: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
) -> float:
    """Estimate the total flux by fitting a clipped 2D Gaussian.

    The fit follows the workflow used in ``UHDR.ipynb``: a clipped detector
    image is compared to a Gaussian model with fixed shape parameters, and only
    the total flux is optimized.
    """

    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return float(np.sum(image))

    clipped_data = image.ravel()
    dummy_x = np.arange(clipped_data.size, dtype=float)

    def model(_, flux):
        amplitude = _flux_to_amplitude(flux, sigma_x, sigma_y)
        modeled = dark_level + _gaussian_2d(grid_x, grid_y, x0, y0, sigma_x, sigma_y, amplitude, theta)
        modeled = np.clip(modeled, 0.0, saturation)
        return (modeled - dark_level).ravel()

    initial_flux = max(float(np.sum(image)), 1.0)

    try:
        popt, _ = curve_fit(model, dummy_x, clipped_data, p0=(initial_flux,), bounds=(0.0, np.inf), maxfev=20000)
        return float(popt[0])
    except Exception:
        return float(np.sum(image))


def _acquire_single_pixel_jit(
    psi: np.ndarray,
    e: float,
    ideal: bool = False,
    qe: float = 1.0,
    dc: float = 0.0,
    ron: float = 0.0,
    fwc: float = math.inf,
) -> int:
    """Numba-accelerated scalar detector model used for the 1-pixel fast path."""

    expected_pe = np.sum(np.abs(psi) ** 2) * e * qe
    expected_dc = dc * e
    total_expected = expected_pe + expected_dc

    if ideal:
        detected_electrons = int(total_expected)
    else:
        if total_expected <= 2000000000.0:
            detected_electrons = np.random.poisson(total_expected)
        else:
            detected_electrons = int(total_expected + np.random.normal(0.0, math.sqrt(total_expected)))

        if ron > 0.0:
            detected_electrons = int(detected_electrons + np.random.normal(0.0, ron))

    if detected_electrons < 0:
        detected_electrons = 0
    if detected_electrons > fwc:
        detected_electrons = int(fwc)

    return detected_electrons


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
        resolution (int): Detector side length in pixels. Default: 1.
        sigma_x (float | None): Gaussian width along x in pixels. Defaults to
            ``resolution / 10``.
        sigma_y (float | None): Gaussian width along y in pixels. Defaults to
            ``resolution / 10``.
        x0 (float): Gaussian center along x in pixels, relative to the grid
            center. Default: 0.0.
        y0 (float): Gaussian center along y in pixels, relative to the grid
            center. Default: 0.0.
        theta (float): Gaussian rotation angle in radians. Default: 0.0.
        uhdr (bool): If ``True`` and the detector saturates, fit a clipped 2D
            Gaussian to recover the total flux.

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
        '_resolution',
        '_sigma_x',
        '_sigma_y',
        '_x0',
        '_y0',
        '_theta',
        '_uhdr',
        '_grid_x',
        '_grid_y',
    )

    def __init__(
        self,
        e: u.Quantity = None,
        ideal: bool = False,
        name: str = 'Unnamed Camera',
        ron: float = 0.0,
        dc: float = 0.0,
        fwc: float = math.inf,
        qe: float = 1.0,
        resolution: int = 1,
        sigma_x: float | None = None,
        sigma_y: float | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        theta: float = 0.0,
        uhdr: bool = False,
    ):
        self._parent_interferometer = None
        self.resolution = resolution
        self.sigma_x = self.resolution / 10 if sigma_x is None else sigma_x
        self.sigma_y = self.resolution / 10 if sigma_y is None else sigma_y
        self.x0 = x0
        self.y0 = y0
        self.theta = theta
        self.uhdr = uhdr
        self.e = e if e is not None else 1 * u.s
        self.ideal = ideal
        self.name = name
        self.ron = ron
        self.dc = dc
        self.fwc = fwc
        self.qe = qe

    def __str__(self) -> str:
        res = f'Camera "{self.name}"\n'
        res += f'  Exposure time: {self.e:.2f}\n'
        res += f'  QE: {self.qe * 100:.1f}% | RON: {self.ron} e- | DC: {self.dc} e-/s\n'
        res += f'  Resolution: {self.resolution} px | UHDR: {self.uhdr}\n'
        if self.resolution > 1:
            res += (
                f'  Gaussian PSF: sigma_x={self.sigma_x:.2f} px | sigma_y={self.sigma_y:.2f} px | '
                f'x0={self.x0:.2f} px | y0={self.y0:.2f} px | theta={self.theta:.2f} rad'
            )
        return res

    def __repr__(self) -> str:
        return self.__str__()

    def _update_grid_cache(self):
        """Cache the centered pixel coordinate grid used in UHDR mode."""

        coordinates = np.arange(self._resolution, dtype=float) - self._resolution // 2
        self._grid_x, self._grid_y = np.meshgrid(coordinates, coordinates)

    def _simulate_spatial_frame(self, signal_flux: float) -> tuple[np.ndarray, bool]:
        """Simulate a multi-pixel detector frame and report saturation."""

        amplitude = _flux_to_amplitude(signal_flux, self.sigma_x, self.sigma_y)
        signal = _gaussian_2d(self._grid_x, self._grid_y, self.x0, self.y0, self.sigma_x, self.sigma_y, amplitude, self.theta)
        raw_frame = signal + self.dc * self._e

        if not self.ideal:
            raw_frame = np.random.poisson(np.clip(raw_frame, 0.0, None)).astype(float)
            if self.ron > 0.0:
                raw_frame = raw_frame + np.random.normal(0.0, self.ron, size=raw_frame.shape)

        saturated = bool(np.any(raw_frame >= self.fwc))
        clipped_frame = np.clip(raw_frame, 0.0, self.fwc)
        return clipped_frame - self.dc * self._e, saturated

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
        self._e_unit = e.unit
        self._e = e_val

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

    @property
    def dc(self) -> float:
        """Dark current in electrons/pixel/second."""

        return self._dc

    @dc.setter
    def dc(self, val: float):
        self._dc = float(val)

    @property
    def fwc(self) -> float:
        """Full-well capacity in electrons."""

        return self._fwc

    @fwc.setter
    def fwc(self, val: float):
        self._fwc = float(val)

    @property
    def qe(self) -> float:
        """Quantum efficiency in the range [0, 1]."""

        return self._qe

    @qe.setter
    def qe(self, val: float):
        self._qe = float(val)

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
        self._update_grid_cache()

    @property
    def sigma_x(self) -> float:
        """Gaussian width along x in pixels."""

        return self._sigma_x

    @sigma_x.setter
    def sigma_x(self, value: float):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError('sigma_x must be a float') from exc
        if value <= 0:
            raise ValueError('sigma_x must be positive')
        self._sigma_x = value

    @property
    def sigma_y(self) -> float:
        """Gaussian width along y in pixels."""

        return self._sigma_y

    @sigma_y.setter
    def sigma_y(self, value: float):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError('sigma_y must be a float') from exc
        if value <= 0:
            raise ValueError('sigma_y must be positive')
        self._sigma_y = value

    @property
    def x0(self) -> float:
        """Gaussian center along x in pixels."""

        return self._x0

    @x0.setter
    def x0(self, value: float):
        try:
            self._x0 = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError('x0 must be a float') from exc

    @property
    def y0(self) -> float:
        """Gaussian center along y in pixels."""

        return self._y0

    @y0.setter
    def y0(self, value: float):
        try:
            self._y0 = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError('y0 must be a float') from exc

    @property
    def theta(self) -> float:
        """Gaussian rotation angle in radians."""

        return self._theta

    @theta.setter
    def theta(self, value: float):
        try:
            self._theta = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError('theta must be a float') from exc

    @property
    def uhdr(self) -> bool:
        """Whether the camera should reconstruct saturated multi-pixel frames."""

        return self._uhdr

    @uhdr.setter
    def uhdr(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError('uhdr must be a boolean')
        self._uhdr = value

    def acquire(self, psi: np.ndarray[complex]) -> int:
        """Simulate an observation on the detector.

        The fast path keeps the historical 1-pixel detector model. When
        ``resolution > 1``, the signal is spread over a centered Gaussian PSF
        on a ``resolution x resolution`` grid, then clipped by saturation and
        optionally reconstructed through a UHDR-style Gaussian fit.

        Args:
            psi (np.ndarray[complex]): Complex electric field amplitudes.

        Returns:
            int: Detected electrons for the exposure.
        """

        if self.resolution == 1:
            return _acquire_single_pixel_jit(psi, self._e, ideal=self._ideal, qe=self._qe, dc=self._dc, ron=self._ron, fwc=self._fwc)

        total_signal = float(np.sum(np.abs(psi) ** 2) * self._e * self._qe)
        frame, saturated = self._simulate_spatial_frame(total_signal)

        if self.uhdr and saturated:
            estimated_flux = _estimate_flux_from_clipped_image(
                frame,
                dark_level=self.dc * self._e,
                saturation=self.fwc,
                grid_x=self._grid_x,
                grid_y=self._grid_y,
                x0=self.x0,
                y0=self.y0,
                sigma_x=self.sigma_x,
                sigma_y=self.sigma_y,
                theta=self.theta,
            )
            return int(max(0.0, np.rint(estimated_flux)))

        return int(max(0.0, np.rint(np.sum(frame))))
