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

    def test_take(self, cmd):
        return cmd + "res = M.take(a, ind)"

    def test_take_ary_mth(self, cmd):
        return cmd + "res = a.take(ind)"

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
            VAL = "val = R.random(ind.size, np.float64, bohrium=BH); "
            yield cmd + VAL
            yield cmd + "ind = ind[::2]; " + VAL
            if shape[0] > 2:
                yield cmd + "ind = ind[1:];" + VAL
            if len(shape) > 1 and shape[1] > 5:
                yield cmd + "ind = ind[3:];" + VAL

    def test_put(self, cmd):
        return cmd + "M.put(res, ind, val)"

    def test_put_ary_mth(self, cmd):
        return cmd + "res.put(ind, val)"

    def test_indexing(self, cmd):
        return cmd + "res = res.flatten(); res[ind] = val"

    def test_cond(self, cmd):
        cmd += cmd + "mask = R.random(val.size, np.bool, bohrium=BH).reshape(val.shape); "
        np_cmd = cmd + "np.put(res, ind[mask], val[mask])"
        bh_cmd = cmd + "M.cond_scatter(res, ind, val, mask)"
        return (np_cmd, bh_cmd)
