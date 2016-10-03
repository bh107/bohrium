import util


class test_array_create:
    def init(self):
        for t in util.TYPES.ALL:
            yield t

    def test_zeros(self, dtype):
        cmd = "res = M.zeros(%d,dtype=%s)" % (100, dtype)
        return cmd

    def test_ones(self, dtype):
        cmd = "res = M.ones(%d,dtype=%s)" % (100, dtype)
        return cmd

    def test_random(self, dtype):
        cmd = "R = bh.random.RandomState(42); res = R.random(%d,dtype=%s, bohrium=BH)" % (100, dtype)
        return cmd