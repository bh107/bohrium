
class test_double_write:
    def init(self):
        yield ""

    def test_double_write(self, _):
        return """
def apply_op(p1):
    cf = M.arange(10*2).reshape(10, 2)
    P1 = M.empty((10, 2))
    P1[:,0] = p1[1:10+1]
    P1[:,1] = p1[2:10+2]
    res = M.zeros(10)
    res[...] = M.add.reduce(cf * P1, axis=1)
    return res

p1 = M.arange(10+4)
res = apply_op(p1)"""



class test_fallback2numpy:
    def init(self):
        yield ""

    def test_bh2np(self, _):
        bh_cmp = """
import gc
gc.disable()
bh.flush()
a = bh.ones(10)
b = np.array(a)
res = np.array(b)
"""
        np_cmd = "res = np.ones(10)"
        return (np_cmd, bh_cmp)
