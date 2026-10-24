# Checks ParallelContext's Python-object collectives on any number of ranks:
# py_alltoall, py_allgather, py_gather, py_broadcast and py_scatter.
#
# Run it as, for example,
#   mpiexec -n 3 python test/parallel_tests/test_py_collectives.py
#   mpiexec -n 3 nrniv -mpi -python test/parallel_tests/test_py_collectives.py
# Both forms use the pure-Python interface's implementation of these methods;
# the second checks that it is also used, and works, when nrniv starts Python.
#
# MPI is started with NEURON_INIT_MPI=1 (set below if unset), so the file also
# runs under plain `python` with one rank.

import os

os.environ.setdefault("NEURON_INIT_MPI", "1")

from neuron import h

pc = h.ParallelContext()
myid = int(pc.id())
nhost = int(pc.nhost())


def item(rank):
    """A different, non-trivial Python object for each rank."""
    return {"rank": rank, "values": list(range(rank + 1)), "name": f"r{rank}"}


def test_allgather():
    assert pc.py_allgather(item(myid)) == [item(i) for i in range(nhost)]
    # Types survive the trip: ints stay ints, None stays None.
    assert pc.py_allgather(myid) == list(range(nhost))
    assert pc.py_allgather(None) == [None] * nhost


def test_alltoall():
    # Rank i sends (i, j) to rank j, so rank j gets [(0, j), (1, j), ...].
    result = pc.py_alltoall([(myid, j) for j in range(nhost)])
    assert result == [(i, myid) for i in range(nhost)], result
    # None items send nothing and arrive as None.
    sent = [None if j % 2 else item(myid) for j in range(nhost)]
    expected = [None if myid % 2 else item(i) for i in range(nhost)]
    assert pc.py_alltoall(sent) == expected
    # Messages of very different sizes.
    big = pc.py_alltoall([list(range(1000 * (myid + 1)))] * nhost)
    assert big == [list(range(1000 * (i + 1))) for i in range(nhost)]


def test_alltoall_sizes_only():
    # A negative second argument returns (bytes sent, bytes received)
    # without exchanging the data. With one rank there is nothing to
    # exchange and NEURON returns the list itself, so this needs 2 or more.
    if nhost == 1:
        return
    sent, received = pc.py_alltoall([b"x" * 10] * nhost, -1)
    assert sent > 0 and received > 0
    assert pc.py_alltoall([None] * nhost, -1) == (0, 0)


def test_gather():
    for root in range(nhost):
        result = pc.py_gather(item(myid), root)
        if myid == root:
            assert result == [item(i) for i in range(nhost)]
        else:
            assert result is None


def test_broadcast():
    for root in range(nhost):
        mine = item(myid)
        result = pc.py_broadcast(mine, root)
        assert result == item(root)
    assert pc.py_broadcast(None, 0) is None


def test_scatter():
    for root in range(nhost):
        source = [item(100 + j) for j in range(nhost)] if myid == root else None
        assert pc.py_scatter(source, root) == item(100 + myid)
    source = [None] * nhost if myid == 0 else None
    assert pc.py_scatter(source, 0) is None


def test_vectors():
    # HOC Vectors are pickled and rebuilt on the receiving rank.
    # Needs the Vector pickling fix (branch fix-vector-pickle), which is being
    # merged separately; fails on 2 or more ranks until then.
    vectors = pc.py_allgather(h.Vector([myid, myid + 0.5]))
    assert [list(v) for v in vectors] == [[i, i + 0.5] for i in range(nhost)]


def test_wrong_list_size():
    # Raised before any message is sent, on every rank, so no rank hangs.
    # (The exception type and text differ between NEURON's C++ version and
    # the Python one, so only check that it is an error.)
    try:
        pc.py_alltoall([0] * (nhost + 1))
    except Exception:
        pass
    else:
        raise AssertionError("expected an error for a list of the wrong size")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            pc.barrier()
    if myid == 0:
        print(f"test_py_collectives: all passed on {nhost} rank(s)", flush=True)
    pc.barrier()
    h.quit()
