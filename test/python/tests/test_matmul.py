import util


class test_matmul:
    def init(self):
        for t in ['np.float32','np.int64','np.complex128']:
            maxdim = 4
            for m in range(2,maxdim+1)[::-1]:
                for n in range(1,maxdim+1)[::-1]:
                    for k in range(2,maxdim+1)[::-1]:
                        cmd = "R = bh.random.RandomState(42);  "
                        cmd += "a = R.random((%d,%d), %s, bohrium=BH); "%(m,k,t)
                        cmd += "b = R.random((%d,%d), %s, bohrium=BH); "%(k,n,t)
                        yield cmd

    def test_matmul(self, cmd):
        cmd += "res = bh.matmul(a, b)"
        return cmd

    def test_dot(self, cmd):
        cmd += "res = M.dot(a , b)"
        return cmd
