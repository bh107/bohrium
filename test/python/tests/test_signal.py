import util


class test_1d:
    def init(self):
        for mode in ['same', 'valid', 'full']:
            for dtype in util.TYPES.FLOAT:
                for cmd1, shape1 in util.gen_random_arrays("R", 1, min_ndim=1, samples_in_each_ndim=1, dtype=dtype):
                    for cmd2, shape2 in util.gen_random_arrays("R", 1, min_ndim=1, samples_in_each_ndim=1, dtype=dtype):
                        if shape1[0] > 0 and shape2[0] > 0:
                            cmd = "R = bh.random.RandomState(42); a=%s; v=%s;" % (cmd1, cmd2)
                            yield (cmd, mode)

    def test_correlate(self, args):
        (cmd, mode) = args
        cmd += "res = M.correlate(a, v, mode='%s')" % mode
        return cmd

    def test_convolve(self, args):
        (cmd, mode) = args
        cmd += "res = M.convolve(a, v, mode='%s')" % mode
        return cmd


class _test_scipy:
    def init(self):
        for mode in ['valid', 'full', 'same']:
            for dtype in util.TYPES.FLOAT:
                for cmd1, shape1 in util.gen_random_arrays("R", 3, min_ndim=1, max_dim=10, samples_in_each_ndim=1,
                                                           dtype=dtype, no_views=True):
                    if util.prod(shape1) <= 0:
                        continue
                    # Notice, the second shape must have the same number of dimension as the first one
                    # and the max dimension cannot be larger than in the first shape
                    # Finally, dimensions above 5 simple take too long
                    max_dim = min(min(shape1), 5)
                    for cmd2, shape2 in util.gen_random_arrays("R", len(shape1), min_ndim=len(shape1), max_dim=max_dim,
                                                               samples_in_each_ndim=1, dtype=dtype, no_views=True):
                        if util.prod(shape2) > 0:
                            cmd = "R = bh.random.RandomState(42); a=%s; v=%s;" % (cmd1, cmd2)
                            yield (cmd, mode)

    def test_correlate(self, args):
        (cmd, mode) = args
        scipy_cmd = cmd + "from scipy import signal; res = signal.correlate(a, v, mode='%s')" % mode
        bh_cmd = cmd + "res = M.correlate_scipy(a, v, mode='%s')" % mode
        return (scipy_cmd, bh_cmd)

    def test_convolve(self, args):
        (cmd, mode) = args
        scipy_cmd = cmd + "from scipy import signal; res = signal.convolve(a, v, mode='%s')" % mode
        bh_cmd = cmd + "res = bh.convolve_scipy(a, v, mode='%s')" % mode
        return (scipy_cmd, bh_cmd)

try:
    import scipy
    test_scipy = _test_scipy
except ImportError:
    print("SciPy not found, skipping the multidimensional tests of convolve() and correlate()")
