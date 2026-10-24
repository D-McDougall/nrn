"""Lazy HOC documentation data, without importing NEURON's Python binding."""
import importlib.util
import io
import pickle
import types
import zlib
from pathlib import Path

_HELP = None
_BUNDLED_HELP = Path(__file__).with_name("_data") / "help_data.dat"


class _HelpUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # The upstream format is a compressed dict[str, str], not executable
        # objects. Refuse GLOBAL/REDUCE constructors even from installed data.
        raise pickle.UnpicklingError("documentation cannot load Python globals")


def _help_paths():
    try:
        spec = importlib.util.find_spec("neuron")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        for directory in spec.submodule_search_locations or ():
            yield Path(directory) / "help_data.dat"
    yield _BUNDLED_HELP


def _load_help():
    global _HELP
    # Concurrent first reads may duplicate this read-only load; publishing a
    # fully validated dictionary is atomic, so that benign race is tolerated.
    if _HELP is None:
        for path in _help_paths():
            try:
                raw = zlib.decompress(path.read_bytes())
                data = _HelpUnpickler(io.BytesIO(raw)).load()
                if type(data) is not dict or not all(
                    type(key) is str and type(value) is str
                    for key, value in data.items()
                ):
                    continue
            except Exception:
                continue
            _HELP = data
            break
        else:
            # Help must not disable otherwise working HOC dispatch when a
            # source checkout or third-party package omits documentation data.
            _HELP = {}
    return _HELP


def get_doc(key, fallback=""):
    """Return the installed core's help text, or the bundled/fallback text."""
    return _load_help().get(key, fallback)


class _ClassDoc:
    """Non-data descriptor: class creation and ordinary use do no help I/O."""

    def __init__(self, name, fallback=None):
        self.name = name
        self.fallback = fallback or f"NEURON HOC class: {name}"

    def __get__(self, instance, owner=None):
        return get_doc(self.name, self.fallback)


class HocDocMethod:
    """A Python method whose __doc__ is NEURON's help text for ``key``.

    The text is read on first lookup, so class creation does no help I/O.
    """

    def __init__(self, key, func):
        self._key = key
        self._func = func
        self._doc_loaded = False

    def __get__(self, instance, owner=None):
        func = self._func
        if not self._doc_loaded:
            func.__doc__ = get_doc(self._key, func.__doc__)
            self._doc_loaded = True
        return func if instance is None else types.MethodType(func, instance)


def hoc_doc(key):
    """Decorator: give a Python method NEURON's help text for ``key``."""
    return lambda func: HocDocMethod(key, func)


class _TopLevelDoc:
    """``h.__doc__``: the neuron.h overview from neuron.doc, read on use."""

    def __get__(self, instance, owner=None):
        try:
            from ..doc import doc_h
        except ImportError:
            return None
        return doc_h
