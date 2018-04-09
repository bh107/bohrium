import util


class test_random_nontrivial:
    def init(self):
        cmd_bh = "R = M.random.RandomState(42); "
        for shape in [10, (10,), (10, 11)]:
            cmd_np = "res = np.zeros(%s, dtype=np.bool); " % repr(shape)
            cmd_np += "res.flat[0] = True; "
            for dtype in util.TYPES.FLOAT:
                yield cmd_np, cmd_bh, shape, dtype

    def test_random(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        cmd_bh += "a = R.random(%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh

    def test_rand(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        if isinstance(shape, int):
            shape = (shape,)
        cmd_bh += "a = R.rand(*%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh

    def test_standard_normal(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        cmd_bh += "a = R.standard_normal(%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh

    def test_randn(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        if isinstance(shape, int):
            shape = (shape,)
        cmd_bh += "a = R.randn(*%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh

    def test_standard_exponential(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        cmd_bh += "a = R.standard_exponential(%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh

    def test_randint(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        cmd_bh += "a = R.randint(1000, size=%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh

    def test_random_integers(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        cmd_bh += "a = R.random_integers(1000, size=%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh

    def test_uniform(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        cmd_bh += "a = R.uniform(0, 10, size=%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh

    def test_normal(self, arg):
        cmd_np, cmd_bh, shape, dtype = arg
        cmd_bh += "a = R.normal(0, 10, size=%s, dtype=%s); " % (shape, dtype)
        cmd_bh += "res = a == a.flatten()[0]"
        return cmd_np, cmd_bh
