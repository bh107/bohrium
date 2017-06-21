import util


class test_vector:
    def init(self):
        for mode in ['same', 'valid', 'full']:
            for dtype in ['np.float32']:  # float32 fails because we are using another algorithm than NumPy
                for cmd1, shape1 in util.gen_random_arrays("R", 1, min_ndim=1, samples_in_each_ndim=1, dtype=dtype):
                    for cmd2, shape2 in util.gen_random_arrays("R", 1, min_ndim=1, samples_in_each_ndim=1, dtype=dtype):
                        if shape1[0] > 0 and shape2[0] > 0:
                            cmd = "R = bh.random.RandomState(42); a=%s; v=%s;" % (cmd1, cmd2)
                            yield (cmd, mode)

    def test_correlate(self, args):
        (cmd, mode) = args
        cmd += "res = M.correlate(a, v, mode='%s')" % mode
        return cmd
