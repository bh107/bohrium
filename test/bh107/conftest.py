import os


def pytest_sessionstart(session):
    os.environ.update(
        # activate __array_function__ interface
        NUMPY_EXPERIMENTAL_ARRAY_FUNCTION='1',

        # prevent some precision issues
        BH_OPENMP_VOLATILE='true',
        BH_OPENCL_VOLATILE='true',
        BH_CUDA_VOLATILE='true',

        # do not flood us with warnings
        BH107_ON_NUMPY_FALLBACK='ignore'
    )
