import numpy as np
import bohrium as bh
from numpytest import numpytest,gen_views,TYPES

class test_bohrium(numpytest):

    def init(self):
        for v in gen_views(1,2):
            a = {}
            exec(v)
            yield (a,v)

    def test_tally(self,a):
        cmd = "\n".join([
            "res = a[0] + a[0]",
            'if "target" in np.__dict__ and "tally" in np.target.__dict__:',
            "    np.target.tally()"
        ])
        exec(cmd)
        return (res,cmd)
