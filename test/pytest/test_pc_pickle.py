# ParallelContext moving Python objects through the bulletin board in one
# process: pc.submit with a Python callable, pc.pyret, pc.post and pc.pack of
# Python objects, and pc.upkpyobj. pc.runworker() returns at once without MPI,
# and the jobs run inside pc.working().

import pytest

from neuron import h

pc = h.ParallelContext()


def pyobjcnt():
    return h.List("PythonObject").count()


def run_jobs():
    """Run every submitted job and return {userid: pyret()}."""
    results = {}
    while True:
        id = pc.working()
        if id == 0:
            break
        results[pc.userid()] = pc.pyret()
    return results


def echo(*args):
    return [a.to_python() if hasattr(a, "to_python") else a for a in args]


def square(x):
    return (x, x * x)


def nothing():
    return {"called": True}


def raises(msg):
    raise ValueError(msg)


def returns_lambda():
    return lambda: 0


@pytest.fixture(autouse=True)
def worker():
    pc.runworker()
    yield
    pc.done()


def test_submit_argument_types():
    nested = {"a": [1, (2.5, "x")], "b": None}
    pc.submit(1, echo, 2.0)
    pc.submit(2, echo, "text")
    pc.submit(3, echo, h.Vector([1, 2, 3]))
    pc.submit(4, echo, [1, "a"])
    pc.submit(5, echo, nested)
    pc.submit(6, echo, 1.5, "s", h.Vector([4.0]), (1, (2, 3)))
    results = run_jobs()
    assert results == {
        1: [2.0],
        2: ["text"],
        3: [[1.0, 2.0, 3.0]],
        4: [[1, "a"]],
        5: [nested],
        6: [1.5, "s", [4.0], (1, (2, 3))],
    }


def test_submit_result():
    pc.submit(square, 2.0)
    assert list(run_jobs().values()) == [(2.0, 4.0)]


def test_submit_no_arguments():
    pc.submit(nothing)
    assert list(run_jobs().values()) == [{"called": True}]


def test_post_take_upkpyobj():
    value = {"k": [1, "a", (2.0, None)]}
    pc.post("pickle_post", value, [3])
    assert pc.look_take("pickle_post")
    assert pc.upkpyobj() == value
    assert pc.upkpyobj() == [3]


def test_pack_upkpyobj():
    pc.pack([1, "a"], 2.0, {"x": (1, 2)})
    pc.post("pickle_pack")
    pc.take("pickle_pack")
    assert pc.upkpyobj() == [1, "a"]
    assert pc.upkscalar() == 2.0
    assert pc.upkpyobj() == {"x": (1, 2)}


def test_submitted_function_raises():
    pc.submit(raises, "original message")
    # The interface raises the ValueError itself; NEURON's own hooks raise a
    # RuntimeError whose message holds its traceback.
    with pytest.raises(Exception, match="original message"):
        run_jobs()
    run_jobs()
    # The bulletin board still runs the next job.
    pc.submit(square, 3.0)
    assert list(run_jobs().values()) == [(3.0, 9.0)]


def test_unpicklable_result():
    pc.submit(returns_lambda)
    with pytest.raises(Exception, match="pickle"):
        run_jobs()
    run_jobs()
    pc.submit(square, 4.0)
    assert list(run_jobs().values()) == [(4.0, 16.0)]


def test_unpicklable_argument(tmp_path):
    with pytest.raises(Exception, match="pickle"):
        pc.submit(echo, lambda: 0)
    with open(tmp_path / "f.txt", "w") as f:
        with pytest.raises(Exception, match="pickle"):
            pc.post("pickle_file", f)
    pc.submit(square, 5.0)
    assert list(run_jobs().values()) == [(5.0, 25.0)]
    pc.post("pickle_after", [5])
    pc.take("pickle_after")
    assert pc.upkpyobj() == [5]


def test_hoc_object_method():
    h(
        """
begintemplate PcPickleSquare
public sq
func sq() { return $1 * $1 }
endtemplate PcPickleSquare
"""
    )
    obj = h.PcPickleSquare()
    pc.submit(obj, "sq", 7)
    assert pc.working() != 0
    assert pc.retval() == 49
    assert pc.working() == 0


def test_no_pythonobject_leak():
    start = pyobjcnt()
    for i in range(50):
        pc.submit(echo, i, str(i), h.Vector(i), [i, {"i": i}])
    results = run_jobs()
    assert len(results) == 50
    for i in range(50):
        pc.post(f"leak{i}", [i])
    for i in range(50):
        pc.take(f"leak{i}")
        assert pc.upkpyobj() == [i]
    del results
    assert pyobjcnt() == start
