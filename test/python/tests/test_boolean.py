import util
import bohrium


class test_mix_types:
    def init(self):
        for dtype in bohrium._info.numpy_types:
            dtype = "np.%s"%dtype.name
            for cmd, shape in util.gen_random_arrays("R", 1, min_ndim=1, samples_in_each_ndim=1,
                                                     dtype=dtype, no_views=True):
                cmd = "R = bh.random.RandomState(42); res=%s; " % cmd
                yield cmd

    def test_assign(self, cmd):
        return cmd + "res[...] = False"

    def test_add(self, cmd):
        return cmd + "res = res + False"

    def test_mul(self, cmd):
        return cmd + "res = res * True"

