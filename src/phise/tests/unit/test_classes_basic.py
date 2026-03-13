import numpy as np
import astropy.units as u
import pytest

from phise.classes.companion import Companion
from phise.classes.camera import Camera
from phise.classes.telescope import Telescope, get_VLTI_UTs
from phise.classes.target import Target
from phise.classes.chip import Chip
from phise.classes.interferometer import Interferometer


def test_companion_valid_and_invalid():
    """Test Companion constructor validation.

    Verifies that:
    - valid inputs create a Companion with correct types/units
    - invalid types or negative contrast raise appropriate errors
    """
    c = Companion(c=1e-4, ρ=5 * u.mas, θ=0.2 * u.rad, name='pc')
    assert isinstance(c.c, float)
    assert c.name == 'pc'

    with pytest.raises(TypeError):
        Companion(c='nope', ρ=5 * u.mas, θ=0.2 * u.rad)

    with pytest.raises(ValueError):
        Companion(c=-0.1, ρ=5 * u.mas, θ=0.2 * u.rad)


def test_camera_properties_and_acquire():
    """Test Camera property setters and acquire behavior.

    Ensures exposure time and ideal flag behave as expected, that
    invalid assignments raise TypeError, and that acquire returns a
    non-negative integer when the camera is in ideal mode.
    """
    cam = Camera(e=2 * u.s, ideal=True, name='C1')
    assert cam.e == 2 * u.s
    assert cam.ideal is True

    # invalid types
    with pytest.raises(TypeError):
        cam.ideal = 'yes'

    with pytest.raises(TypeError):
        cam.name = 123

    # ideal acquisition deterministic
    np.random.seed(0)
    ψ = np.array([1.0 + 0j, 1.0 + 0j])
    det = cam.acquire(ψ)
    assert isinstance(det, int)
    assert det >= 0


def test_telescope_area_and_position_validations():
    """Validate Telescope area and position setters.

    Confirms correct units and shapes are enforced and that helper
    generator returns expected number of telescopes.
    """
    a = np.pi * (2 * u.m) ** 2
    r = np.array([0.0, 1.0]) * u.m
    tel = Telescope(a=a, r=r, name='Ttest')
    assert tel.a.unit == u.m ** 2
    assert tel.r.unit == u.m

    with pytest.raises(ValueError):
        tel.r = np.array([1.0]) * u.m

    with pytest.raises(TypeError):
        tel.a = 5.0

    # helper function
    v = get_VLTI_UTs()
    assert len(v) == 4


def test_interferometer_basic_setters_and_errors():
    """Test Interferometer construction and basic setter validation.

    Builds a minimal Interferometer and checks that unit conversions
    and input validation for properties raise appropriate errors.
    """
    # minimal valid objects
    tel = Telescope(a=np.pi * (1 * u.m) ** 2, r=np.array([0.0, 0.0]) * u.m)
    chip = Chip()
    cam = Camera(e=1 * u.s, ideal=True)

    # good construction
    intr = Interferometer(l=-24 * u.deg, λ=1.55 * u.um, Δλ=1 * u.nm, fov=10 * u.mas, η=0.5, telescopes=[tel], chip=chip, camera=cam)
    assert intr.l.unit == u.deg

    # wrong types
    with pytest.raises(TypeError):
        intr.l = 10.0

    with pytest.raises(ValueError):
        intr.Δλ = -1 * u.nm
