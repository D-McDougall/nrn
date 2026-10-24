import copy
import gc
import math
import pickle
import struct
import subprocess
import sys

import neuron
from neuron import h

SPECIAL = [-0.0, 1e-300, math.inf, -math.inf, math.nan, 1.5, -2.0]


def bits(seq):
    """Bit patterns of the doubles in seq, so -0.0 and nan compare exactly."""
    return struct.pack(f"{len(seq)}d", *seq)


def sample_vectors():
    return [
        h.Vector(),
        h.Vector([-0.0]),
        h.Vector(SPECIAL + [i * 0.25 for i in range(1000 - len(SPECIAL))]),
    ]


def test_round_trip():
    for v in sample_vectors():
        for p in range(pickle.HIGHEST_PROTOCOL + 1):
            w = pickle.loads(pickle.dumps(v, protocol=p))
            assert len(w) == len(v)
            assert bits(w) == bits(v)
            assert w.hocobjptr() != v.hocobjptr()
            if len(v):
                old = v[0]
                w[0] = 42.0
                assert bits([v[0]]) == bits([old])
                w.resize(len(w) + 1)
                assert len(v) == len(w) - 1


def test_no_shared_ownership():
    v = h.Vector([1, 2, 3])
    w = pickle.loads(pickle.dumps(v))
    del v
    gc.collect()
    assert list(w) == [1, 2, 3]
    w.resize(10)
    assert len(w) == 10
    assert list(w)[:3] == [1, 2, 3]


def test_fresh_process():
    code = (
        "import pickle, sys\n"
        "from neuron import h\n"
        "sys.stdout.flush()\n"
        "data = pickle.dumps(h.Vector([1.5, -0.0, 1e-300]))\n"
        "sys.stdout.buffer.write(data.hex().encode())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True
    ).stdout
    # NEURON may print a banner before the hex digits.
    data = bytes.fromhex(out.split()[-1].decode())
    w = pickle.loads(data)
    assert bits(w) == bits([1.5, -0.0, 1e-300])


def test_reduce_format():
    v = h.Vector([1.5, -0.0, math.inf])
    assert v.__reduce__() == (
        neuron._pkl,
        (0,),
        (1, struct.pack("d", 2.0), len(v), struct.pack(f"{len(v)}d", *v)),
    )
    e = h.Vector()
    assert e.__reduce__() == (neuron._pkl, (0,), (1, struct.pack("d", 2.0), 0, b""))


def test_byte_swap():
    # Use the byte order this machine does not have.
    other = ">" if sys.byteorder == "little" else "<"
    values = [1.5, -0.0, 1e-300, math.inf]
    state = (
        1,
        struct.pack(other + "d", 2.0),
        len(values),
        struct.pack(f"{other}{len(values)}d", *values),
    )
    v = neuron._pkl(0)
    v.__setstate__(state)
    assert bits(v) == bits(values)


def expect(exc_type, message, func, *args):
    try:
        func(*args)
    except exc_type as e:
        assert str(e) == message, str(e)
    else:
        assert False, f"{exc_type.__name__} not raised"


def test_bad_states():
    two = struct.pack("d", 2.0)
    expect(
        ValueError,
        "endian_data size is not sizeof(double)",
        h.Vector().__setstate__,
        (1, two[:4], 1, struct.pack("d", 1.0)),
    )
    expect(
        ValueError,
        "buffer size does not match array size",
        h.Vector().__setstate__,
        (1, two, 2, struct.pack("d", 1.0)),
    )
    expect(
        TypeError,
        "pickle not returning string",
        h.Vector().__setstate__,
        (1, two, 1, "abcdefgh"),
    )
    expect(
        TypeError,
        "pickle not returning string",
        h.Vector().__setstate__,
        (1, "abcdefgh", 1, struct.pack("d", 1.0)),
    )


def test_non_vectors():
    message = "HocObject: Only Vector instance can be pickled"
    sec = h.Section("pickle_soma")
    for obj in [h.List(), h.IClamp(sec(0.5))]:
        for p in range(pickle.HIGHEST_PROTOCOL + 1):
            expect(TypeError, message, pickle.dumps, obj, p)
    expect(TypeError, message, copy.copy, h.List())
    v = h.Vector([1, 2])
    for c in [copy.copy(v), copy.deepcopy(v)]:
        assert c.hocobjptr() != v.hocobjptr()
        c[0] = 5
        assert list(v) == [1, 2]


def test_containers():
    v = h.Vector([1, 2, 3])
    d = pickle.loads(pickle.dumps({"a": v, "b": [v, 1]}))
    assert list(d["a"]) == [1, 2, 3]
    assert d["b"][1] == 1
    assert d["a"] is d["b"][0]
    assert d["a"].hocobjptr() != v.hocobjptr()


def test_master_compatibility():
    # protocol=2 pickle of h.Vector([1.5, -2.0, 3.25]), made with the C++
    # module (NEURON 9.1.0.dev0) on a little-endian machine.
    data = (
        b"\x80\x02cneuron\n_pkl\nq\x00K\x00\x85q\x01Rq\x02(K\x01c_codecs\nencode\n"
        b"q\x03X\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@q\x04X\x06\x00\x00"
        b"\x00latin1q\x05\x86q\x06Rq\x07K\x03h\x03X\x1a\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\xc3\xb8?\x00\x00\x00\x00\x00\x00\x00\xc3\x80\x00\x00\x00"
        b"\x00\x00\x00\n@q\x08h\x05\x86q\tRq\ntq\x0bb."
    )
    v = pickle.loads(data)
    assert list(v) == [1.5, -2.0, 3.25]
    # And the other way: on a little-endian machine, this module's
    # protocol-2 pickle is the same bytes.
    if sys.byteorder == "little":
        assert pickle.dumps(h.Vector([1.5, -2.0, 3.25]), protocol=2) == data


if __name__ == "__main__":
    test_round_trip()
    test_no_shared_ownership()
    test_fresh_process()
    test_reduce_format()
    test_byte_swap()
    test_bad_states()
    test_non_vectors()
    test_containers()
    test_master_compatibility()
