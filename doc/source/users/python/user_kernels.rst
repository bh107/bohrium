UserKernel
~~~~~~~~~~

Bohrium supports user kernel, which makes it possible to implement a specialized handwritten kernel. The idea is that if you encounter a problem that you cannot implement using array programming and Bohrium cannot accelerate, you can write a kernel in C99 that calls other libraries or do the calculation itself.

OpenMP Example
--------------

In order to write and run your own kernel use `bh.user_kernel.execute() <https://github.com/bh107/bohrium/blob/master/bridge/npbackend/bohrium/user_kernel.py#L21>`_::

    import bohrium as bh

    def fftn(ary):
        # Making sure that `ary` is complex, contiguous, and uses no offset
        ary = bh.user_kernel.make_behaving(ary, dtype=bh.complex128)
        res = bh.empty_like(a)

        # Indicates the direction of the transform you are interested in;
        # technically, it is the sign of the exponent in the transform.
        sign = ["FFTW_FORWARD", "FFTW_BACKWARD"]

        kernel = """
        #include <stdint.h>
        #include <stdlib.h>
        #include <complex.h>
        #include <fftw3.h>

        #if defined(_OPENMP)
            #include <omp.h>
        #else
            static inline int omp_get_max_threads() { return 1; }
            static inline int omp_get_thread_num()  { return 0; }
            static inline int omp_get_num_threads() { return 1; }
        #endif

        void execute(double complex *in, double complex *out) {
            const int ndim = %(ndim)d;
            const int shape[] = {%(shape)s};
            const int sign = %(sign)s;

            fftw_init_threads();
            fftw_plan_with_nthreads(omp_get_max_threads());

            fftw_plan p = fftw_plan_dft(ndim, shape, in, out, sign, FFTW_ESTIMATE);
            if(p == NULL) {
                printf("fftw plan fail!\\n");
                exit(-1);
            }
            fftw_execute(p);
            fftw_destroy_plan(p);
            fftw_cleanup_threads();
        }
        """ % {'ndim': a.ndim, 'shape': str(a.shape)[1:-1], 'sign': sign[0]}

        # Adding some extra link options to the compiler command
        cmd = bh.user_kernel.get_default_compiler_command() + " -lfftw3 -lfftw3_threads"
        bh.user_kernel.execute(kernel, [ary, res], compiler_command=cmd)
        return res

OpenCL Example
--------------

In order to use the OpenCL backend, use the `tag` and `param` of `bh.user_kernel.execute() <https://github.com/bh107/bohrium/blob/master/bridge/npbackend/bohrium/user_kernel.py#L21>`_::

    import bohrium as bh

    kernel = """
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    kernel void execute(global double *a, global double *b) {
        int i0 = get_global_id(0);
        int i1 = get_global_id(1);
        int gid = i0 * 5 + i1;
        b[gid] = a[gid] + gid;
    }
    """
    a = bh.ones(10*5, bh.double).reshape(10,5)
    res = bh.empty_like(a)
    # Notice, the OpenCL backend requires global_work_size and local_work_size
    bh.user_kernel.execute(kernel, [a, res],
                           tag="opencl",
                           param={"global_work_size": [10, 5], "local_work_size": [1, 1]})
    print(res)

.. note:: Remember to use the OpenCL backend by setting `BH_STACK=opencl`.


