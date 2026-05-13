import astropy.units as u
import numpy as np

from phise.modules import roc


class _DummyChip:
    def __init__(self):
        self.nb_raw_outputs = 2
        self.nb_processed_outputs = 1

    def process_outputs(self, raw):
        return np.array([raw[0] - raw[1]])


class _DummyInterferometer:
    def __init__(self):
        self.chip = _DummyChip()
        self.fov = 1.0 * u.mas


class _DummyTarget:
    def __init__(self):
        self.companions = []


class _DummyContext:
    def __init__(self):
        self.interferometer = _DummyInterferometer()
        self.chip = self.interferometer.chip
        self.target = _DummyTarget()

    def observe(self):
        return np.array([1.0, 2.0])


def test_get_gpu_core_count_returns_none_or_positive_int():
    core_count = roc.get_gpu_core_count()
    assert core_count is None or (isinstance(core_count, int) and core_count > 0)


def test_generate_data_parallel_matches_serial_on_deterministic_context():
    ctx_h1 = _DummyContext()
    ctx_h0 = _DummyContext()

    t0_serial, t1_serial = roc.generate_data(
        ctx_h1,
        ctx_h0=ctx_h0,
        nmc=8,
        size=6,
        n_jobs=1,
    )

    t0_parallel, t1_parallel = roc.generate_data(
        ctx_h1,
        ctx_h0=ctx_h0,
        nmc=8,
        size=6,
        n_jobs=4,
    )

    assert np.array_equal(t0_serial, t0_parallel)
    assert np.array_equal(t1_serial, t1_parallel)


def test_generate_data_raises_for_invalid_n_jobs():
    ctx_h1 = _DummyContext()

    try:
        roc.generate_data(ctx_h1, nmc=2, size=2, n_jobs=0)
        raised = False
    except ValueError:
        raised = True

    assert raised
