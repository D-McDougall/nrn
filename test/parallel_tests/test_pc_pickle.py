# Checks that ParallelContext's bulletin board moves Python objects between
# ranks: pc.submit with a Python callable and its arguments, pc.pyret, and
# pc.post / pc.take / pc.upkpyobj.
#
# Run it as, for example,
#   mpiexec -n 3 python test/parallel_tests/test_pc_pickle.py
#   mpiexec -n 3 nrniv -mpi -python test/parallel_tests/test_pc_pickle.py
# Under `nrniv -python` with libnrnpython linked in, NEURON's own pickling
# hooks (nrnpy_p2h.cpp) do the work; otherwise the pure-Python interface's do.
#
# MPI is started with NEURON_INIT_MPI=1 (set below if unset), so the file also
# runs under plain `python` with one rank.

import os

os.environ.setdefault("NEURON_INIT_MPI", "1")

from neuron import h

pc = h.ParallelContext()
nhost = int(pc.nhost())
njobs = 4 * nhost


def job(i, label, vec, obj):
    """Return everything the job got, and the rank that ran it."""
    return {
        "i": i,
        "label": label,
        "vec": vec.to_python(),
        "obj": obj,
        "rank": int(pc.id()),
    }


def expected(i):
    return {
        "i": float(i),
        "label": f"job{i}",
        "vec": [float(i), i + 0.5],
        "obj": {"n": i, "items": [i, (i, "x")]},
    }


def take_posted(key):
    """Take the Python objects posted under key, on whichever rank runs it."""
    pc.take(key)
    return [pc.upkpyobj(), pc.upkpyobj(), int(pc.id())]


def test_submit_pyret():
    for i in range(njobs):
        pc.submit(i + 1, job, i, f"job{i}", h.Vector([i, i + 0.5]), expected(i)["obj"])
    results = {}
    while pc.working():
        results[int(pc.userid())] = pc.pyret()
    assert sorted(results) == list(range(1, njobs + 1)), sorted(results)
    for userid, result in results.items():
        rank = result.pop("rank")
        assert 0 <= rank < nhost
        assert result == expected(userid - 1), result


def test_post_upkpyobj():
    # Rank 0 posts; a job, run by any rank, takes the posting and returns it.
    for i in range(njobs):
        pc.post(f"pickle{i}", {"i": i, "v": [i, "a"]}, (i, None))
    for i in range(njobs):
        pc.submit(i + 1, take_posted, f"pickle{i}")
    results = {}
    while pc.working():
        results[int(pc.userid())] = pc.pyret()
    assert sorted(results) == list(range(1, njobs + 1)), sorted(results)
    for userid, (posted, extra, rank) in results.items():
        i = userid - 1
        assert posted == {"i": i, "v": [i, "a"]}, posted
        assert extra == (i, None), extra
        assert 0 <= rank < nhost


if __name__ == "__main__":
    pc.runworker()  # ranks other than 0 run jobs here until pc.done()
    test_submit_pyret()
    test_post_upkpyobj()
    pc.done()
    print(f"test_pc_pickle: all passed on {nhost} rank(s)", flush=True)
    h.quit()
