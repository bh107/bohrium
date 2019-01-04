import bohrium_api


class _test_openmp:
    def init(self):
        yield ""

    def test_add(self, _):
        bh_cmd = '''
kernel = r"""
#include <stdint.h>
void execute(double *a, double *b, double *c) {
    for(uint64_t i=0; i<100; ++i) {
        c[i] = a[i] + b[i] + i;
    }
}
"""
a = bh.ones(100, bh.double)
b = bh.ones(100, bh.double)
res = bh.empty_like(a)
bh.user_kernel.execute(kernel, [a, b, res])
'''
        np_cmd = '''
a = np.ones(100, bh.double)
b = np.ones(100, bh.double)
res = a + b + np.arange(100)
'''
        return (np_cmd, bh_cmd)


class _test_opencl:
    def init(self):
        yield ""

    def test_add(self, _):
        bh_cmd = '''
kernel = r"""
kernel void execute(global double *a, global double *b, global double *c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i] + i;
}
"""
a = bh.ones(100, bh.double)
b = bh.ones(100, bh.double)
res = bh.empty_like(a)
bh.user_kernel.execute(kernel, [a, b, res], tag="opencl", param="global_work_size: 100; local_work_size: 1")
'''
        np_cmd = '''
a = np.ones(100, bh.double)
b = np.ones(100, bh.double)
res = a + b + np.arange(100)
'''
        return (np_cmd, bh_cmd)


if bohrium_api.stack_info.is_proxy_in_stack():
    print("Skipping test, the proxy backend does not support user kernels")
else:
    test_openmp = _test_openmp
    if bohrium_api.stack_info.is_opencl_in_stack():
        test_opencl = _test_opencl


