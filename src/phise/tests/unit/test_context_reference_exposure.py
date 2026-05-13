import numpy as np
import math
import astropy.units as u
import pytest

from phise.classes.camera import Camera
from phise.classes.chip import Chip
from phise.classes.context import Context
from phise.classes.interferometer import Interferometer
from phise.classes.target import Target
from phise.classes.telescope import Telescope


def _build_minimal_context(camera: Camera) -> Context:
    """Create a minimal physically valid context for exposure-time tests."""

    telescope = Telescope(a=np.pi * (1 * u.m) ** 2, r=np.array([0.0, 0.0]) * u.m)
    interferometer = Interferometer(
        l=-24 * u.deg,
        λ=1.55 * u.um,
        Δλ=200 * u.nm,
        fov=10 * u.mas,
        η=0.4,
        telescopes=[telescope],
        chip=Chip(),
        camera=camera,
    )
    target = Target(
        f=1e-7 * u.W / u.m**2 / u.nm,
        δ=-30 * u.deg,
        companions=[],
    )
    return Context(
        interferometer=interferometer,
        target=target,
        h=0 * u.hourangle,
        Δh=1 * u.hourangle,
        Γ=10 * u.nm,
    )


def test_reference_exposure_time_matches_linear_camera_model():
    """The reference exposure should match the expected linear ADU budget."""

    camera = Camera(
        e=1 * u.s,
        ideal=True,
        qe=0.8,
        gain=2.0,
        dc=10.0,
        fwc=50000,
        resolution=1,
    )
    context = _build_minimal_context(camera)

    exposure = context.get_reference_exposure_time()

    central_pixel_fraction = math.erf(0.5 / (math.sqrt(2.0) * camera.spot_size)) ** 2
    electron_rate = context.pf.sum().to(1 / u.s).value * central_pixel_fraction * camera.qe + camera.dc
    expected_seconds = camera.fwc / electron_rate

    assert exposure.unit == u.s
    assert np.isclose(exposure.to_value(u.s), expected_seconds, rtol=1e-12)


def test_reference_exposure_time_without_dark_current_is_longer():
    """Ignoring dark current should increase the required stellar exposure time."""

    camera = Camera(
        e=1 * u.s,
        ideal=True,
        qe=0.9,
        gain=1.5,
        dc=50.0,
        fwc=60000,
        resolution=3,
    )
    context = _build_minimal_context(camera)

    exposure_with_dark = context.get_reference_exposure_time(include_dark_current=True)
    exposure_without_dark = context.get_reference_exposure_time(include_dark_current=False)

    assert exposure_with_dark > 0 * u.s
    assert exposure_without_dark > exposure_with_dark


def test_reference_exposure_time_increases_with_spot_size():
    """Larger spot sizes should require longer exposure to saturate central pixel."""

    camera_small_spot = Camera(
        e=1 * u.s,
        ideal=True,
        qe=0.9,
        gain=1.0,
        dc=1e-9,
        fwc=30000,
        resolution=11,
        spot_size=0.6,
    )
    camera_large_spot = Camera(
        e=1 * u.s,
        ideal=True,
        qe=0.9,
        gain=1.0,
        dc=1e-9,
        fwc=30000,
        resolution=11,
        spot_size=1.8,
    )

    context_small_spot = _build_minimal_context(camera_small_spot)
    context_large_spot = _build_minimal_context(camera_large_spot)

    exposure_small_spot = context_small_spot.get_reference_exposure_time(include_dark_current=False)
    exposure_large_spot = context_large_spot.get_reference_exposure_time(include_dark_current=False)

    assert exposure_large_spot > exposure_small_spot


def test_reference_exposure_time_rejects_invalid_fraction():
    """The ADU target fraction must stay in (0, 1]."""

    camera = Camera(e=1 * u.s, ideal=True)
    context = _build_minimal_context(camera)

    with pytest.raises(ValueError):
        context.get_reference_exposure_time(target_adu_fraction=0.0)

    assert context.get_reference_exposure_time(target_adu_fraction=1.0) > 0 * u.s

    with pytest.raises(ValueError):
        context.get_reference_exposure_time(target_adu_fraction=1000.0)

    with pytest.raises(TypeError):
        context.get_reference_exposure_time(target_adu_fraction="0.5")
