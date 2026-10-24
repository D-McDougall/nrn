"""obj.__getattribute__(name) must give what getattr(obj, name) gives.

Some user code, such as the ringtest model's ranparm.py, reads mechanism and
point-process parameters with an explicit ``seg.__getattribute__(name)``
call. Each case below builds fresh objects, looks one name up both ways, and
compares the results. A second test checks that a name that does not exist
raises AttributeError, with the same message, both ways.
"""

import neuron
import pytest
from neuron import h, nrn

h(
    """
begintemplate GetattrT
public x, a, s, o, f, p
double a[3]
strdef s
objref o
proc init() {
    x = 7
    a[1] = 2
    s = "hi"
    o = new Vector(2)
    hidden = 5
}
func f() { return 2 }
proc p() { }
endtemplate GetattrT
"""
)
h("create getattr_soma")
h("objref getattr_objref")
h("getattr_objref = new Vector(4)")
h("func getattr_func() { return 3 }")


class PlainSec(nrn.Section):
    pass


class MySec(nrn.Section):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra = 3


class GetattrPy(neuron.hclass(h.GetattrT)):
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        self.py_value = 9


class GetattrPyFallback(neuron.hclass(h.GetattrT)):
    """A subclass whose own __getattr__ calls self.__getattribute__."""

    def __new__(cls):
        return super().__new__(cls)

    def __getattr__(self, name):
        return self.__getattribute__(name)


# Objects that a case's object needs but does not reference itself, such as
# the section of a point process and the source of a NetCon. Kept to the end
# of the module so that NEURON does not free them during the case.
_keep_alive = []


def _section():
    sec = h.Section(name="getattr_sec")
    sec.insert("hh")
    sec.insert("MCna")
    _keep_alive.append(sec)
    return sec


def _seg():
    return _section()(0.5)


def _plainsec():
    sec = PlainSec(name="getattr_plainsec")
    sec.insert("hh")
    return sec


# Python attributes set on a Section subclass are sent to its segments as
# range variables, so MySec cannot be built.
_section_subclass_attribute = pytest.mark.xfail(
    strict=True,
    reason="a Python subclass of Section cannot set its own Python attributes",
)


def _expsyn():
    return h.ExpSyn(_seg())


def _netcon():
    stim = h.NetStim()
    _keep_alive.append(stim)
    return h.NetCon(stim, None)


def _element(i):
    return lambda result: result[i]


# (description, object factory, attribute name, function applied to both
# results before they are compared or None, and optionally a pytest mark)
CASES = [
    # Segment
    ("Segment property v", _seg, "v", None),
    ("Segment property x", _seg, "x", None),
    ("Segment property sec", _seg, "sec", None),
    ("Segment property diam", _seg, "diam", None),
    ("Segment property cm", _seg, "cm", None),
    ("Segment range variable gnabar_hh", _seg, "gnabar_hh", None),
    ("Segment ion variable nao", _seg, "nao", None),
    ("Segment ion variable ena", _seg, "ena", None),
    ("Segment ion variable ina", _seg, "ina", None),
    ("Segment mechanism hh", _seg, "hh", None),
    ("Segment ion mechanism na_ion", _seg, "na_ion", None),
    ("Segment method point_processes", _seg, "point_processes", None),
    ("Segment method area", _seg, "area", None),
    ("Segment reference _ref_v", _seg, "_ref_v", _element(0)),
    ("Segment reference _ref_gnabar_hh", _seg, "_ref_gnabar_hh", _element(0)),
    # Section
    ("Section L", _section, "L", None),
    ("Section nseg", _section, "nseg", None),
    ("Section Ra", _section, "Ra", None),
    ("Section rallbranch", _section, "rallbranch", None),
    ("Section method name", _section, "name", None),
    ("Section method psection", _section, "psection", None),
    ("Section method insert", _section, "insert", None),
    ("Section range variable v", _section, "v", None),
    # Mechanism
    ("Mechanism parameter gnabar", lambda: _seg().hh, "gnabar", None),
    ("Mechanism state m", lambda: _seg().hh, "m", None),
    ("Mechanism method name", lambda: _seg().hh, "name", None),
    ("Mechanism NMODL FUNCTION alp", lambda: _seg().MCna, "alp", None),
    ("ion mechanism variable ena", lambda: _seg().na_ion, "ena", None),
    # Point processes and artificial cells
    ("ExpSyn tau", _expsyn, "tau", None),
    ("ExpSyn i", _expsyn, "i", None),
    ("ExpSyn get_loc", _expsyn, "get_loc", None),
    ("ExpSyn has_loc", _expsyn, "has_loc", None),
    ("ExpSyn hname", _expsyn, "hname", None),
    ("NetStim interval", h.NetStim, "interval", None),
    # Built-in HOC objects
    ("Vector size", lambda: h.Vector(3), "size", None),
    ("Vector x", lambda: h.Vector([4, 5, 6]), "x", _element(1)),
    ("List count", h.List, "count", None),
    ("NetCon weight", _netcon, "weight", _element(0)),
    ("NetCon delay", _netcon, "delay", None),
    ("Random uniform", h.Random, "uniform", None),
    ("ParallelContext nhost", h.ParallelContext, "nhost", None),
    ("SectionRef sec", lambda: h.SectionRef(sec=_section()), "sec", None),
    ("SectionList append", h.SectionList, "append", None),
    ("MechanismStandard count", lambda: h.MechanismStandard("hh"), "count", None),
    # HOC templates
    ("template public scalar", h.GetattrT, "x", None),
    ("template public array", h.GetattrT, "a", None),
    ("template public array element", h.GetattrT, "a", _element(1)),
    ("template strdef", h.GetattrT, "s", None),
    ("template objref", h.GetattrT, "o", None),
    ("template func", h.GetattrT, "f", None),
    ("template proc", h.GetattrT, "p", None),
    ("template non-public member", h.GetattrT, "hidden", None),
    # Python subclasses
    ("Section subclass NEURON attribute L", _plainsec, "L", None),
    ("Section subclass range variable v", _plainsec, "v", None),
    (
        "Section subclass Python attribute",
        lambda: MySec(name="getattr_mysec"),
        "extra",
        None,
        _section_subclass_attribute,
    ),
    ("hclass subclass Python attribute", GetattrPy, "py_value", None),
    ("hclass subclass template member", GetattrPy, "x", None),
    ("hclass subclass with __getattr__", GetattrPyFallback, "x", None),
    # h
    ("h global t", lambda: h, "t", None),
    ("h global dt", lambda: h, "dt", None),
    ("h global celsius", lambda: h, "celsius", None),
    ("h mechanism GLOBAL minf_hh", lambda: h, "minf_hh", None),
    ("h built-in function finitialize", lambda: h, "finitialize", None),
    ("h class Vector", lambda: h, "Vector", None),
    ("h class ExpSyn", lambda: h, "ExpSyn", None),
    ("h template name", lambda: h, "GetattrT", None),
    ("h section created in HOC", lambda: h, "getattr_soma", None),
    ("h objref", lambda: h, "getattr_objref", None),
    ("h HOC func", lambda: h, "getattr_func", None),
]


def _same(a, b):
    """Whether two lookups of one name gave the same NEURON value."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if a == b:
        return True
    # Mechanisms, mechanism functions and bound methods do not define ==.
    # Their str names the mechanism or function, and for a bound method also
    # the object it is bound to.
    return str(a) == str(b)


@pytest.mark.parametrize(
    "factory, name, after",
    [pytest.param(*case[1:4], id=case[0], marks=case[4:]) for case in CASES],
)
def test_getattribute_matches_getattr(factory, name, after):
    obj = factory()
    explicit = obj.__getattribute__(name)
    normal = getattr(obj, name)
    if after is not None:
        explicit = after(explicit)
        normal = after(normal)
    assert _same(explicit, normal), (explicit, normal)


MISSING_CASES = [
    ("Segment", _seg, "no_such_name"),
    ("Section", _section, "no_such_name"),
    ("Mechanism", lambda: _seg().hh, "no_such_name"),
    ("ion mechanism", lambda: _seg().na_ion, "no_such_name"),
    ("ExpSyn", _expsyn, "no_such_name"),
    ("NetStim", h.NetStim, "no_such_name"),
    ("Vector", h.Vector, "no_such_name"),
    ("SectionRef", lambda: h.SectionRef(sec=_section()), "no_such_name"),
    ("template", h.GetattrT, "no_such_name"),
    ("Section subclass", _plainsec, "no_such_name"),
    ("hclass subclass", GetattrPy, "no_such_name"),
    ("hclass subclass with __getattr__", GetattrPyFallback, "no_such_name"),
    ("h", lambda: h, "no_such_name"),
]


@pytest.mark.parametrize(
    "factory, name",
    [pytest.param(*case[1:], id=case[0]) for case in MISSING_CASES],
)
def test_missing_name_raises(factory, name):
    obj = factory()
    with pytest.raises(AttributeError) as explicit:
        obj.__getattribute__(name)
    with pytest.raises(AttributeError) as normal:
        getattr(obj, name)
    assert str(explicit.value) == str(normal.value)
