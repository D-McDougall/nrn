"""Startup options the legacy extension's PyInit_hoc passed to ivocmain.

Used when this package is NEURON's own interface (``neuron._interface``), so
``from neuron import h`` starts NEURON the way the compiled module did
(src/nrnpython/inithoc.cpp:261-380): ``__main__.neuron_options``,
``NEURON_MODULE_OPTIONS``, ``-mpi`` (from ``NEURON_INIT_MPI`` or an MPI that
mpi4py already started), the mechanism library NEURON's tests rely on being
loaded from ``<arch>/`` in the working directory, and ``NEURON_NFRAME``.
Standalone myneuron keeps its own minimal argv.
"""

import os
import platform
import sys


def _add(args, name, value=None):
    # Native add_arg skips a name already present anywhere in argv.
    if name in args:
        return
    args.append(name)
    if value is not None:
        args.append(value)


def _main_options(args, write):
    """Add ``__main__.neuron_options``; return True for -print-options."""
    options = getattr(sys.modules.get("__main__"), "neuron_options", None)
    if options is None:
        return False
    if not isinstance(options, dict):
        write("__main__.neuron_options is not a dict\n")
        return False
    print_options = False
    for key, value in options.items():
        if not isinstance(key, (str, bytes)) or not (
            value is None or isinstance(value, (str, bytes))
        ):
            write("A neuron_options key:value is not a string:string or string:None\n")
            continue
        key = key.decode() if isinstance(key, bytes) else key
        value = value.decode() if isinstance(value, bytes) else value
        if key == "-print-options":
            print_options = True
            continue
        _add(args, key, value)
    return print_options


def split_options(text):
    """Port of add_space_separated_options: escapes, '...' and "..."."""
    tokens, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        token = []
        while i < n and not text[i].isspace():
            char = text[i]
            i += 1
            if char == "\\" and i < n and (text[i].isspace() or text[i] in "\"'"):
                token.append(text[i])
                i += 1
            elif char in "\"'":
                while i < n and text[i] != char:
                    if text[i] == "\\" and i + 1 < n and text[i + 1] == char:
                        i += 1
                    token.append(text[i])
                    i += 1
                i += 1  # closing quote (or end of string)
            else:
                token.append(char)
        tokens.append("".join(token))
    return tokens


def mechanism_library(cwd="."):
    """Relative path of the auto-loaded mechanism library, or None."""
    if sys.platform == "win32":
        prefix, suffix = "", ".dll"
    elif sys.platform == "darwin":
        prefix, suffix = "lib", ".dylib"
    else:
        prefix, suffix = "lib", ".so"
    path = f"{platform.machine()}/{prefix}nrnmech{suffix}"
    return path if os.path.isfile(os.path.join(cwd, path)) else None


def _mpi_initialized_by_mpi4py():
    """True if mpi4py has already called MPI_Init in this process.

    Only looks at a module that is already imported. Importing mpi4py.MPI
    here would itself start MPI, which must stay the user's decision.
    """
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is None:
        return False
    try:
        return bool(mpi.Is_initialized()) and not mpi.Is_finalized()
    except Exception:
        # A partly imported or unusual mpi4py: behave as if MPI is not running.
        return False


def mpi_requested(args, environ, mpi_enabled):
    """Should NEURON start (or join) MPI? Port of PyInit_hoc's MPI block.

    The rules follow src/nrnpython/inithoc.cpp:278-343:

    * ``-mpi`` already in the options (from ``neuron_options`` or
      ``NEURON_MODULE_OPTIONS``) is always kept. In a build without MPI,
      libnrniv then prints its usual warning, as it did before.
    * In a build without MPI, ``NEURON_INIT_MPI`` is ignored silently.
    * ``NEURON_INIT_MPI=1`` asks for MPI.
    * ``NEURON_INIT_MPI=0`` turns off the automatic check below.
    * Otherwise MPI is used when it is already running, which in practice
      means the script did ``from mpi4py import MPI`` before importing
      neuron. Master asks MPI_Initialized() directly. Here we ask mpi4py,
      so that no MPI library has to be loaded just to find out.

    Nothing else is needed: ``-mpi`` makes nrn_init call nrn_mpi_setup
    (src/ivoc/ivocmain.cpp), which loads libmpi in dynamic-MPI builds and
    calls nrnmpi_init. nrnmpi_init skips MPI_Init when MPI is already
    initialized.
    """
    if "-mpi" in args:
        return True
    if not mpi_enabled:
        return False
    env_mpi = environ.get("NEURON_INIT_MPI")
    if env_mpi == "1":
        return True
    if env_mpi == "0":
        return False
    return _mpi_initialized_by_mpi4py()


def _build_has_mpi():
    """True if this NEURON was built with MPI (NRN_ENABLE_MPI=ON)."""
    try:
        from .._config_params import mpi_enabled
    except ImportError:
        # Standalone myneuron, or a build tree older than this setting.
        return False
    return mpi_enabled


def startup_options(write=sys.stdout.write, environ=os.environ, mpi_enabled=None):
    """Arguments after argv[0], in the order PyInit_hoc adds them.

    ``mpi_enabled`` defaults to what the build reports. Tests pass it to
    check the MPI rules without needing an MPI build.
    """
    if mpi_enabled is None:
        mpi_enabled = _build_has_mpi()
    args = []
    print_options = _main_options(args, write)
    for token in split_options(environ.get("NEURON_MODULE_OPTIONS", "")):
        if token == "-print-options":
            print_options = True
        else:
            _add(args, token)
    if mpi_requested(args, environ, mpi_enabled):
        _add(args, "-mpi")
    library = mechanism_library()
    if library:
        _add(args, "-dll", library)
    nframe = environ.get("NEURON_NFRAME")
    if nframe is not None:
        if nframe.isdigit() and int(nframe) > 0:
            _add(args, "-NFRAME", nframe)
        elif nframe.lstrip("-").isdigit():
            write("NEURON_NFRAME env value must be positive\n")
        else:
            write("NEURON_NFRAME env value is invalid!\n")
    if print_options:
        write("ivocmain options:" + "".join(f" '{arg}'" for arg in args) + "\n")
    return args
