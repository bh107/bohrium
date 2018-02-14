import util

class test_array:
    def init(self):
        for t in util.TYPES.ALL:
            yield t

    def test_list(self, dtype):
        cmd = "res = M.array([1,2,3], dtype=%s)" % dtype
        return cmd

    def test_scalar(self, dtype):
        cmd = "res = M.array(42, dtype=%s)" % dtype
        return cmd

    def test_tuple_nest(self, dtype):
        cmd = "res = M.array((42, np.array([43]), M.array([44])), dtype=%s)" % dtype
        return cmd

class test_array_create:
    def init(self):
        for t in util.TYPES.ALL:
            yield t

    def test_zeros(self, dtype):
        cmd = "res = M.zeros(%d, dtype=%s)" % (100, dtype)
        return cmd

    def test_ones(self, dtype):
        cmd = "res = M.ones(%d, dtype=%s)" % (100, dtype)
        return cmd

    def test_random(self, dtype):
        cmd = "R = bh.random.RandomState(42); res = R.random(%d, dtype=%s, bohrium=BH)" % (100, dtype)
        return cmd

    def test_copy(self, dtype):
        return self.test_random(dtype) + "; res = res.copy()"

class test_arange:
    def init(self):
        for start in range(-3, 3):
            for stop in range(-1, 6):
                for step in range(1, 4):
                    yield (start, stop, step)

        for start in range(3, -3, -1):
            for stop in range(5, -3, -1):
                for step in range(-3, 0):
                    yield (start, stop, step)

    def test_arange(self, args):
        (start, stop, step) = args
        return "res = M.arange(%d, %d, %d, dtype=np.float64)" % (start, stop, step)

class test_seq_of_scalars:
    def init(self):
        yield ""

    def test_list_of_scalars(self, cmd):
        return "res = M.array(list(map(M.array, range(10))))"
