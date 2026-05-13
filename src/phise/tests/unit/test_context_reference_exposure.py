import numpy as np
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
        max_adu=50000,
        resolution=1,
    )
    context = _build_minimal_context(camera)

    exposure = context.get_reference_exposure_time()

    electron_rate = context.pf.sum().to(1 / u.s).value * camera.qe + camera.dc
    expected_seconds = (0.5 * camera.max_adu * camera.gain) / electron_rate

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
        max_adu=60000,
        resolution=3,
    )
    context = _build_minimal_context(camera)

    exposure_with_dark = context.get_reference_exposure_time(include_dark_current=True)
    exposure_without_dark = context.get_reference_exposure_time(include_dark_current=False)

    assert exposure_with_dark > 0 * u.s
    assert exposure_without_dark > exposure_with_dark


def test_reference_exposure_time_rejects_invalid_fraction():
    """The ADU target fraction must stay strictly inside (0, 1)."""

    camera = Camera(e=1 * u.s, ideal=True)
    context = _build_minimal_context(camera)

    with pytest.raises(ValueError):
        context.get_reference_exposure_time(target_adu_fraction=0.0)

    with pytest.raises(ValueError):
        context.get_reference_exposure_time(target_adu_fraction=1.0)

    with pytest.raises(TypeError):
        context.get_reference_exposure_time(target_adu_fraction="0.5")
