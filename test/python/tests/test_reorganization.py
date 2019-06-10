import util
import functools
import operator


class test_gather:
    def init(self):
        for ary, shape in util.gen_random_arrays("R", 3, max_dim=50, dtype="np.float64"):
            nelem = functools.reduce(operator.mul, shape)
            if nelem == 0:
                continue
            cmd = "R = bh.random.RandomState(42); a = %s; " % ary
            cmd += "ind = M.arange(%d, dtype=np.int64).reshape(%s); " % (nelem, shape)
            yield cmd
            yield cmd + "ind = ind[::2]; "
            if shape[0] > 2:
                yield cmd + "ind = ind[1:]; "
            if len(shape) > 1 and shape[1] > 5:
                yield cmd + "ind = ind[3:]; "

    @util.add_bh107_cmd
    def test_take(self, cmd):
        return cmd + "res = M.take(a, ind)"

    def test_take_ary_mth(self, cmd):
        return cmd + "res = a.take(ind)"

    @util.add_bh107_cmd
    def test_indexing(self, cmd):
        return cmd + "res = a.flatten()[ind.flatten()]"


class test_scatter:
    def init(self):
        for ary, shape in util.gen_random_arrays("R", 3, max_dim=50, dtype="np.float64"):
            nelem = functools.reduce(operator.mul, shape)
            if nelem == 0:
                continue
            cmd = "R = bh.random.RandomState(42); res = %s; " % ary
            cmd += "ind = M.arange(%d, dtype=np.int64).reshape(%s); " % (nelem, shape)
            VAL = "val = R.random(shape=ind.shape, bohrium=BH).astype(np.float64); "
            yield cmd + VAL
            yield cmd + "ind = ind[::2]; " + VAL
            if shape[0] > 2:
                yield cmd + "ind = ind[1:];" + VAL
            if len(shape) > 1 and shape[1] > 5:
                yield cmd + "ind = ind[3:];" + VAL

    @util.add_bh107_cmd
    def test_put(self, cmd):
        return cmd + "M.put(res, ind, val)"

    @util.add_bh107_cmd
    def test_put_scalar(self, cmd):
        return cmd + "M.put(res, ind, 42)"

    @util.add_bh107_cmd
    def test_put_fixed_length_val(self, cmd):
        return cmd + "M.put(res, ind, M.arange(10))"

    def test_put_ary_mth(self, cmd):
        return cmd + "res.put(ind, val)"

    @util.add_bh107_cmd
    def test_indexing(self, cmd):
        return cmd + "res = res.flatten(); res[ind] = val"

    def test_cond(self, cmd):
        cmd += cmd + "mask = R.random(shape=ind.size, bohrium=BH).astype(np.bool).reshape(ind.shape); "
        np_cmd = cmd + "np.put(res, ind[mask], val[mask])"
        bh_cmd = cmd + "M.cond_scatter(res, ind, val, mask)"
        bh107_cmd = bh_cmd.replace("bh.random.RandomState", "bh107.random.RandomState").replace(", bohrium=BH", "") \
            .replace("bh.take", "bh107.take")
        return (np_cmd, bh_cmd, bh107_cmd)


class test_nonzero:
    def init(self):
        for ary, shape in util.gen_random_arrays("R", 3, max_dim=50, dtype="np.float64"):
            nelem = functools.reduce(operator.mul, shape)
            if nelem == 0:
                continue
            cmd = "R = bh.random.RandomState(42); a = %s; " % ary
            yield cmd

    @util.add_bh107_cmd
    def test_flatnonzero(self, cmd):
        return cmd + "res = M.flatnonzero(a)"

    def test_nonzero(self, cmd):
        return cmd + "res = M.concatenate(M.nonzero(a))"


class test_fancy_indexing_get:
    def init(self):
        for ary, shape in util.gen_random_arrays("R", 3, max_dim=50, dtype="np.float64"):
            nelem = functools.reduce(operator.mul, shape)
            if nelem == 0:
                continue
            cmd = "R = bh.random.RandomState(42); a = %s; " % ary
            ind = "ind = ("
            for dim in shape:
                ind += "R.random(shape=(10,), bohrium=BH).astype(np.uint64) %% %d, " % dim
            ind += "); "
            yield cmd + ind

    def test_take_using_index_tuple(self, cmd):
        cmd += "res = bh.take_using_index_tuple(a, ind)"
        bh107_cmd = cmd.replace("bh.random.RandomState",
                                "bh107.random.RandomState") \
            .replace(", bohrium=BH", "") \
            .replace("bh.take", "bh107.take")
        return (cmd, cmd, bh107_cmd)

    @util.add_bh107_cmd
    def test_indexing(self, cmd):
        return cmd + "res = a[ind]"


class test_fancy_indexing_set:
    def init(self):
        for ary, shape in util.gen_random_arrays("R", 3, max_dim=50, dtype="np.float64"):
            nelem = functools.reduce(operator.mul, shape)
            if nelem == 0:
                continue
            cmd = "R = bh.random.RandomState(42); res = %s; " % ary
            ind = "ind = ("
            for dim in shape:
                ind += "R.random(shape=(10,), bohrium=BH).astype(np.uint64) %% %d, " % dim
            ind += "); "
            yield cmd + ind

    def test_put_using_index_tuple(self, cmd):
        cmd += "bh.put_using_index_tuple(res, ind, 42)"
        bh107_cmd = cmd.replace("bh.random.RandomState",
                                "bh107.random.RandomState") \
            .replace(", bohrium=BH", "") \
            .replace("bh.put", "bh107.put")
        return (cmd, cmd, bh107_cmd)

    @util.add_bh107_cmd
    def test_indexing(self, cmd):
        return cmd + "res[ind] = 42"
