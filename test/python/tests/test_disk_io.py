import util

class test_save:
    def init(self):
        for t in util.TYPES.ALL:
            cmd = """
from tempfile import NamedTemporaryFile
f = NamedTemporaryFile()
R = bh.random.RandomState(42)
M.save(f, R.random(100, dtype=%s, bohrium=BH))
f.flush()
import os
os.fsync(f.fileno())
""" % t
            yield cmd

    def test_load(self, cmd):
        return cmd + "res = M.load(f.name)"


class test_savetxt:
    def init(self):
        for t in util.TYPES.ALL_INT + util.TYPES.FLOAT:
            cmd = """
from tempfile import NamedTemporaryFile
f = NamedTemporaryFile()
R = bh.random.RandomState(42)
M.savetxt(f, R.random(100, dtype=%s, bohrium=BH))
f.flush()
import os
os.fsync(f.fileno())
""" % t
            yield cmd

    def test_loadtxt(self, cmd):
        return cmd + "res = M.loadtxt(f.name)"
