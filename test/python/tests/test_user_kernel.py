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
a = bh.ones(100, np.double)
b = bh.ones(110, np.double)
b = bh.user_kernel.make_behaving(b[10:])
res = bh.empty_like(a)
bh.user_kernel.execute(kernel, [a, b, res])
'''
        np_cmd = '''
a = np.ones(100, np.double)
b = np.ones(110, np.double)
b = b[10:]
res = a + b + np.arange(100)
'''
        return (np_cmd, bh_cmd, bh_cmd.replace("bh.", "bh107."))


class _test_opencl:
    def init(self):
        yield ""

    def test_add(self, _):
        bh_cmd = '''
kernel = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void execute(global double *a, global double *b, global double *c) {
    int i0 = get_global_id(0);
    int i1 = get_global_id(1);
    int gid = i0 * 5 + i1;
    c[gid] = a[gid] + b[gid] + gid;
}
"""
a = bh.ones((20, 5), bh.double)
b = bh.ones((21, 5), bh.double)
b = bh.user_kernel.make_behaving(b[1:, :])
res = bh.empty_like(a)
bh.user_kernel.execute(kernel, [a, b, res], tag="opencl", param={"global_work_size": (20, 5), "local_work_size": (1, 1)})
'''
        np_cmd = '''
a = np.ones((20, 5), bh.double)
b = np.ones((21, 5), bh.double)
b = b[1:, :]
res = a + b + np.arange(100).reshape(20, 5)
'''
        return (np_cmd, bh_cmd)


if bohrium_api.stack_info.is_proxy_in_stack():
    print("Skipping test, the proxy backend does not support user kernels")
else:
    test_openmp = _test_openmp
    if bohrium_api.stack_info.is_opencl_in_stack():
        test_opencl = _test_opencl


