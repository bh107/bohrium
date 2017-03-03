import util


class test_complex_views:
    def init(self):
        for dtype in util.TYPES.COMPLEX:
            for cmd, shape in util.gen_random_arrays("R", 2, dtype=dtype):
                cmd = "R = bh.random.RandomState(42); z = %s; " % cmd
                yield cmd


    def test_real_imag(self, cmd):
        return cmd + "res = z.real + z.imag"

    def test_real(self, cmd):
        return cmd + "res = z + z.real"

    def test_imag(self, cmd):
        return cmd + "res = z + z.imag"


