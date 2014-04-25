import bohrium as np
from numpytest import numpytest

class test_matmul(numpytest):
    def init(self):
        print "matmul isn't supported by the new bridge"
        raise StopIteration()
        self.config['maxerror'] = 0.00001
        for t in ['np.float32','np.float64','np.int64']:
            maxdim = 6
            for m in range(1,maxdim+1):
                for n in range(1,maxdim+1):
                    for k in range(1,maxdim+1):
                        a = {}
                        cmd  = "a[0] = self.array((%d,%d),%s);"%(m,k,t)
                        cmd += "a[1] = self.array((%d,%d),%s);"%(k,n,t)
                        exec cmd
                        yield (a,cmd)

    def test_matmul(self,a):
        cmd = "a[2] = np.matmul(a[0],a[1])"
        exec cmd
        return (a[2], cmd)

    def test_dot(self,a):
        cmd = "a[2] = np.dot(a[0],a[1])"
        exec cmd
        return (a[2], cmd)
