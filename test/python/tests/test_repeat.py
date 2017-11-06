class test_repeat:
    """ Test repeat"""
    def init(self):
        cmd = "a = M.arange(10);"
        yield (cmd)

    def test_lowlevel(self, cmd):
        """Low level test without using any Python syntax suga r"""
        bh_cmd = cmd + """
def kernel(a, b):
    b += a * b
res = M.ones_like(a)
bh.flush()
kernel(a, res)
bh.flush(nrepeats=5)
"""
        np_cmd = cmd + """
def kernel(a, b):
    b += a * b
res = M.ones_like(a)
for _ in range(5):
    kernel(a, res)
"""
        return (np_cmd, bh_cmd)
