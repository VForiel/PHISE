import numpy as np
import astropy.units as u

from phise.classes.camera import Camera


def test_take_dark_bypasses_previous_dark_subtraction():
    """Dark estimation must not depend on any previously stored master dark.

    This regression test ensures ``take_dark`` computes a physical dark level
    even when ``_dark`` contains a large stale value.
    """

    camera = Camera(
        e=1 * u.s,
        ideal=False,
        ron=0.0,
        dc=1000.0,
        gain=1.0,
        resolution=1,
        max_adu=1_000_000_000,
    )

    camera._dark = np.array([[500_000.0]])
    estimated_dark = camera.take_dark(N=200, e=0.01 * u.s)

    mean_dark = float(np.mean(estimated_dark))

    # Expected dark level is around dc * e = 10 e-/ADU for this setup.
    assert mean_dark > 0.0
    assert mean_dark < 1000.0
