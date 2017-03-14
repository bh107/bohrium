import sys

NP_AS_BH = "import sys; sys.modules['numpy'] = bh; "

class _test_scipy:
    def init(self):
        yield (NP_AS_BH)

    def test_scpy(self, cmd):
        cmd += "import scipy; res = M.ones(10)"
        return cmd

    def test_sparse(self, cmd):
        cmd += "import scipy.sparse; res = M.ones(10)"
        return cmd

    def test_io(self, cmd):
        cmd += "import scipy.io; res = M.ones(10)"
        return cmd

try:
    import scipy
    test_scipy = _test_scipy
except ImportError:
    print("SciPy not found, ignoring some tests")


class _test_matplotlib:
    def init(self):
        yield (NP_AS_BH)

    def test_matplotlib(self, cmd):
        cmd += "import matplotlib; res = M.ones(10)"
        return cmd

    def test_pyplot(self, cmd):
        cmd += "import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt; " \
              "res = M.ones(10).reshape((2,5)); plt.imshow(res)"
        return cmd

try:
    if sys.version_info[0] >= 3:
        print("Matplotlib not supported in Python 3")
    else :
        import matplotlib
        test_matplotlib = _test_matplotlib
except ImportError:
    print("Matplotlib not found, ignoring some tests")


class _test_netCDF4:
    def init(self):
        yield (NP_AS_BH)

    def test_netCDF4(self, cmd):
        cmd += "import netCDF4; res = M.ones(10)"
        return cmd

    def test_Dataset(self, cmd):
        cmd += "from netCDF4 import Dataset; res = M.ones(10);"
        return cmd

try:
    import netCDF4
    test_netCDF4 = _test_netCDF4
except ImportError:
    print("netCDF4 not found, ignoring some tests")

