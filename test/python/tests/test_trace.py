import util

class test_trace:
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 4, min_ndim=2, dtype="np.uint32"):
            cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
            yield cmd

    def test_regular(self, cmd):
        cmd += "res = M.trace(a)"
        return cmd


class test_trace_offset:
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 4, min_ndim=2, dtype="np.uint32"):
            cmd = "R = bh.random.RandomState(42); a = %s" % cmd
            for offset in range(shape[0], shape[0]+1):
                yield "%s; offset=%d; " % (cmd, offset)

    def test_trace_offset(self, cmd):
        return cmd + "res = M.trace(a, offset=offset)"


class test_trace_axis:
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 4, min_ndim=2, dtype="np.uint32"):
            cmd = "R = bh.random.RandomState(42); a = %s;" % cmd
            for offset in range(shape[0], shape[0]+1):
                for axis1 in range(len(shape)):
                    for axis2 in range(len(shape)):
                        if axis1 == axis2:
                            continue
                        yield (cmd, offset, axis1, axis2)

    def test_trace_offset(self, args):
        (cmd, offset, axis1, axis2) = args
        return cmd + "res = M.trace(a, offset=%d, axis1=%d, axis2=%d)" % (offset, axis1, axis2)
