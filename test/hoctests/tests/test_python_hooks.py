"""SaveState and RangeVarPlot hooks that call back into Python.

Both work through function pointers in libnrniv. NEURON's own Python module
filled them from libnrnpython; the Python interface now fills them itself,
so they also work when libnrnpython is not loaded (dynamic-Python builds).
"""

import contextlib
import sys
import uuid

import pytest

import neuron
from neuron import h


@contextlib.contextmanager
def savestate_plugin(store, restore):
    """Register a SaveState plugin for the duration of a test."""
    id_ = str(uuid.uuid4())
    neuron.register_savestate(id_, store, restore)
    try:
        yield
    finally:
        i = neuron._id_savestates.index(id_)
        del neuron._id_savestates[i]
        del neuron._store_savestates[i]
        del neuron._restore_savestates[i]


@pytest.fixture
def soma():
    sec = h.Section("soma")
    sec.nseg = 5
    yield sec


def test_savestate_keeps_python_data(soma):
    state = {"value": b"before"}

    def restore(data):
        state["value"] = bytes(data)

    with savestate_plugin(lambda: state["value"], restore):
        h.finitialize(-65)
        ss = h.SaveState()
        ss.save()
        state["value"] = b"changed"
        ss.restore()
    assert state["value"] == b"before"


def test_savestate_python_data_survives_file(soma, tmp_path):
    state = {"value": b"from file"}

    def restore(data):
        state["value"] = bytes(data)

    path = str(tmp_path / "state.dat")
    with savestate_plugin(lambda: state["value"], restore):
        h.finitialize(-65)
        ss = h.SaveState()
        ss.save()
        f = h.File(path)
        f.wopen()
        ss.fwrite(f)
        f.close()

        state["value"] = b"changed"
        ss2 = h.SaveState()
        f = h.File(path)
        f.ropen()
        ss2.fread(f)
        f.close()
        ss2.restore()
    assert state["value"] == b"from file"


def test_savestate_store_error_is_raised(soma):
    def store():
        raise RuntimeError("store failed")

    with savestate_plugin(store, lambda data: None):
        h.finitialize(-65)
        ss = h.SaveState()
        with pytest.raises(RuntimeError, match="store failed"):
            ss.save()


def test_rangevarplot_python_callable(soma):
    h.finitialize(-65)
    rvp = h.RangeVarPlot(lambda x: 2.0 * x, soma(0), soma(1))
    y = h.Vector()
    d = h.Vector()
    rvp.to_vector(y, d)
    assert len(y) == soma.nseg + 2
    for yi, di in zip(y, d):
        assert yi == pytest.approx(2.0 * di / soma.L)


class _Marker:
    """Stands in for an rxd species, which RangeVarPlot cannot call directly."""


def test_rangevarplot_converts_object(soma):
    def to_callable(obj):
        if isinstance(obj, _Marker):
            return lambda x: 7.0
        return obj

    if not neuron._interface.install_rvp_rxd_hook(to_callable):
        pytest.skip("NEURON's own PythonObject provider owns this hook")
    try:
        h.finitialize(-65)
        rvp = h.RangeVarPlot(_Marker(), soma(0), soma(1))
        y = h.Vector()
        rvp.to_vector(y)
        assert list(y) == [7.0] * (soma.nseg + 2)
    finally:
        neuron._interface.install_rvp_rxd_hook(neuron._nrnpy_rvp_pyobj_callback)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
