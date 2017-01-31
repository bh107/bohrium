import util

class test_typecast:
    def init(self):
        for t1 in util.TYPES.ALL:
            for t2 in util.TYPES.ALL:
                yield (t1, t2)

    def test_typecast(self, arg):
        (t1, t2) = arg
        cmd = "R = bh.random.RandomState(42); "
        cmd += "a0 = R.random(10, dtype=%s, bohrium=BH); " % t1
        cmd += "a1 = R.random(10, dtype=%s, bohrium=BH); " % t2
        cmd += "res = a0 + a1"
        return cmd
