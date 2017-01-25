#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#define CL_EXT_PREFIX__VERSION_1_1_DEPRECATED
#define CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
#else
#include <CL/cl.h>
#endif
#undef CL_VERSION_1_2
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
