# Checks the rules that decide whether `import neuron` starts MPI
# (neuron/_interface/_startup.py, mpi_requested and startup_options).
#
# These tests only look at the argument list that would be passed to
# nrn_init. They never start MPI, so they run in every build, with or
# without MPI. The tests in the mpi_init group of test/CMakeLists.txt
# check that MPI really starts.

import sys
import types

from neuron._interface._startup import mpi_requested, startup_options


class FakeMPI(types.ModuleType):
    """Stands in for mpi4py.MPI, with the two calls _startup.py makes."""

    def __init__(self, initialized, finalized=False):
        super().__init__("mpi4py.MPI")
        self._initialized = initialized
        self._finalized = finalized

    def Is_initialized(self):
        return self._initialized

    def Is_finalized(self):
        return self._finalized


def with_fake_mpi4py(module, check):
    """Run check() while sys.modules["mpi4py.MPI"] is `module` (or absent)."""
    saved = sys.modules.pop("mpi4py.MPI", None)
    try:
        if module is not None:
            sys.modules["mpi4py.MPI"] = module
        check()
    finally:
        sys.modules.pop("mpi4py.MPI", None)
        if saved is not None:
            sys.modules["mpi4py.MPI"] = saved


def test_env_var_in_mpi_build():
    assert mpi_requested([], {"NEURON_INIT_MPI": "1"}, mpi_enabled=True)
    assert not mpi_requested([], {"NEURON_INIT_MPI": "0"}, mpi_enabled=True)
    # Unset, and nothing has started MPI: no MPI.
    with_fake_mpi4py(None, lambda: _assert_not_requested({}, True))


def _assert_not_requested(environ, mpi_enabled):
    assert not mpi_requested([], environ, mpi_enabled)


def test_env_var_ignored_without_mpi():
    # Like master: a build without MPI ignores NEURON_INIT_MPI silently.
    assert not mpi_requested([], {"NEURON_INIT_MPI": "1"}, mpi_enabled=False)


def test_explicit_option_is_kept():
    # -mpi given by the user is passed on in every build. Without MPI,
    # libnrniv prints its usual warning, as on master.
    assert mpi_requested(["-mpi"], {}, mpi_enabled=True)
    assert mpi_requested(["-mpi"], {}, mpi_enabled=False)
    assert mpi_requested(["-mpi"], {"NEURON_INIT_MPI": "0"}, mpi_enabled=True)


def test_mpi_started_by_mpi4py():
    def check():
        assert mpi_requested([], {}, mpi_enabled=True)
        # NEURON_INIT_MPI=0 turns the automatic check off.
        assert not mpi_requested([], {"NEURON_INIT_MPI": "0"}, mpi_enabled=True)
        # A build without MPI cannot join.
        assert not mpi_requested([], {}, mpi_enabled=False)

    with_fake_mpi4py(FakeMPI(initialized=True), check)
    # mpi4py imported but MPI not initialized, or already finalized.
    with_fake_mpi4py(
        FakeMPI(initialized=False), lambda: _assert_not_requested({}, True)
    )
    with_fake_mpi4py(
        FakeMPI(initialized=True, finalized=True),
        lambda: _assert_not_requested({}, True),
    )


def test_startup_options_adds_mpi_once():
    def check():
        args = startup_options(
            environ={"NEURON_INIT_MPI": "1", "NEURON_MODULE_OPTIONS": "-mpi"},
            mpi_enabled=True,
        )
        assert args.count("-mpi") == 1, args
        args = startup_options(environ={"NEURON_INIT_MPI": "1"}, mpi_enabled=False)
        assert "-mpi" not in args, args

    with_fake_mpi4py(None, check)


if __name__ == "__main__":
    test_env_var_in_mpi_build()
    test_env_var_ignored_without_mpi()
    test_explicit_option_is_kept()
    test_mpi_started_by_mpi4py()
    test_startup_options_adds_mpi_once()
    print("test_mpi_startup_options: all passed")
