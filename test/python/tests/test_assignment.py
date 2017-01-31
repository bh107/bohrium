import util

class test_assign_vector:
    def init(self):
        yield "res = M.arange(100).reshape(10,10); "

    def test_row(self, cmd):
        return cmd + "res[2, :] = M.ones(10)"

    def test_column(self, cmd):
        return cmd + "res[:, 2] = M.ones(10)"

    def test_row_list(self, cmd):
        return cmd + "res[2, :] = [42]*10"

    def test_column_list(self, cmd):
        return cmd + "res[:, 2] = [42]*10"
