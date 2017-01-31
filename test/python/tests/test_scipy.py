import util

class _test_import:
    """ Test reduction of all kind of views"""
    def init(self):
        yield ("")

    def test_scpy(self, arg):
        cmd = "import scipy; res = M.ones(10)"
        return cmd

    def test_sparse(self, arg):
        cmd = "import scipy.sparse; res = M.ones(10)"
        return cmd

try:
    import scipy
    test_import = _test_import
except ImportError:
    print("SciPy not found, ignoring some tests")
