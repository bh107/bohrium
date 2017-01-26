import util

class test_get:
    def init(self):
        yield "a = M.arange(100).reshape(10,10); "

    def test_row(self, cmd):
        return cmd + "res = a[1, :]"

    def test_column(self, cmd):
        return cmd + "res = a[:, 1]"

    def test_row2(self, cmd):
        return cmd + "res = a[1]"

    def test_scalar(self, cmd):
        return cmd + "res = a[1, 2]"

    def test_scalar_slice(self, cmd):
        return cmd + "res = a[1:2, 1]"
