from numpytest import numpytest, TYPES


class test_array_create:
    def init(self):
        for t in TYPES.NORMAL:
            yield t

    def test_zeros(self, dtype):
        cmd = "res = M.zeros(%d,dtype=%s)" % (100, dtype)
        return cmd

    def test_ones(self, dtype):
        cmd = "res = M.ones(%d,dtype=%s)" % (100, dtype)
        return cmd