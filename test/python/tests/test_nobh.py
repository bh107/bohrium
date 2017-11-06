import util


class test_bincount:
    def init(self):
        for dtype in util.TYPES.SIGNED_INT:
            for shape in util.gen_shapes(1, 100, min_ndim=1, iters=5):
                cmd = "R = bh.random.RandomState(42); "
                cmd += "a=R.random_integers(0, high=100, size=%d, dtype=%s, bohrium=BH);" % (shape[0], dtype)
                yield (cmd)

    def test_bincount(self, cmd):
        cmd += "res = M.bincount(a)"
        return cmd

    def test_int_weights(self, cmd):
        cmd += "w = M.arange(a.shape[0]);"
        cmd += "res = M.bincount(a, w)"
        return cmd

    def test_float_weights(self, cmd):
        cmd += "w = M.arange(a.shape[0]) * 0.42;"
        cmd += "res = M.bincount(a, w)"
        return cmd

