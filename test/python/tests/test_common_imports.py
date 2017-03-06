
class _test_scipy:
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
    test_scipy = _test_scipy
except ImportError:
    print("SciPy not found, ignoring some tests")


class _test_matplotlib:
    def init(self):
        yield ("")

    def test_matplotlib(self, arg):
        cmd = "import matplotlib; res = M.ones(10)"
        return cmd

    def test_pyplot(self, arg):
        cmd = "import matplotlib.pyplot as plt; res = M.ones(10).reshape((2,5)); plt.imshow(res)"
        return cmd

try:
    import matplotlib
    test_matplotlib = _test_matplotlib
except ImportError:
    print("Matplotlib not found, ignoring some tests")


class _test_netCDF4:
    def init(self):
        yield ("")

    def test_netCDF4(self, arg):
        cmd = "import netCDF4; res = M.ones(10)"
        return cmd

    def test_pyplot(self, arg):
        cmd = "import netCDF4.Dataset; res = M.ones(10);"
        return cmd

try:
    import netCDF4
    test_netCDF4 = _test_netCDF4
except ImportError:
    print("netCDF4 not found, ignoring some tests")