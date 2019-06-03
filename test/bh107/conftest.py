import os


def pytest_sessionstart(session):
    os.environ.update(
        NUMPY_EXPERIMENTAL_ARRAY_FUNCTION='1',
        BH_OPENMP_VOLATILE='true',
        BH_OPENCL_VOLATILE='true',
        BH_CUDA_VOLATILE='true',
    )
