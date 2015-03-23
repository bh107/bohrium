#include "thread_control.hpp"

#include <stdlib.h>

#if defined(VE_CPU_BIND)
#include <omp.h>
#include <hwloc.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
inline int omp_get_num_threads() { return 1; }
#endif

using namespace std;

namespace bohrium{
namespace engine{
namespace cpu{

const char ThreadControl::TAG[] = "ThreadControl";

ThreadControl::ThreadControl(thread_binding binding, size_t mthreads)
    : binding_(binding), mthreads_(mthreads)
{
    char* env;

    //
    // Let OpenMP environment vars take precedens.
    //
    // This means the CPU-VE backs off if the "user" insist they
    // know better than the CPU-VE, or they might just wanna
    // experiment :)
    //
    env = getenv("OMP_NUM_THREADS");
    if ((NULL != env) or (0 == mthreads_)) {
        mthreads_ = omp_get_max_threads(); 
    }

    // If GOMP_CPU_AFFINITY is used we let the compiler handle affinity
    env = getenv("GOMP_CPU_AFFINITY");
    if (NULL != env) {
        binding_ = BIND_TO_NONE;
    }

    env = getenv("KMP_AFFINITY");
    if (NULL != env) {
        binding_ = BIND_TO_NONE;
    }
}

ThreadControl::~ThreadControl(void)
{}

string ThreadControl::text(void)
{
    return "Textual representation of thread-control is not implemented";
}

thread_binding ThreadControl::get_binding(void)
{
    return binding_;
}

size_t ThreadControl::get_mthreads(void)
{
    return mthreads_;
}

#if defined(VE_CPU_BIND)
size_t ThreadControl::bind_threads(thread_binding binding)
{
    binding_ = binding;
    if (BIND_TO_NONE == binding) {              // Early exit when not binding
        return 0;                               // EXIT
    }

    hwloc_topology_t topo;
    int obj_count;
    int depth;
    int error = 0;

    hwloc_topology_init(&topo);                 // Setup topology
    hwloc_topology_load(topo);
    if (BIND_TO_PU == binding) {
        depth = hwloc_get_type_depth(topo, HWLOC_OBJ_PU);
    } else {
        depth = hwloc_get_type_depth(topo, HWLOC_OBJ_CORE);
    }
    obj_count = hwloc_get_nbobjs_by_depth(topo, depth);

    if (omp_get_max_threads()>obj_count) {      // Cap the number of threads
        omp_set_num_threads(obj_count);
    }

    #pragma omp parallel num_threads(obj_count)
    {                                           // Bind threads
        int tid = omp_get_thread_num();
        hwloc_obj_t obj = hwloc_get_obj_by_depth(topo, depth, tid);
        if (obj) {
            hwloc_cpuset_t cpuset = hwloc_bitmap_dup(obj->cpuset);
            hwloc_bitmap_singlify(cpuset);
            if (hwloc_set_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD)) {
                error = errno;
            }
            hwloc_bitmap_free(cpuset);
        } else {
            error = 1;
        }
        // NOTE: Don't care about the race on the error, the function
        //       will only return a single 'errno', out of 'obj_count'
        //       potential number of errors anyway, so the race is not
        //       "harmful". However, this could return the amount of
        //       errors instead one arbitrary error.
    }

    hwloc_topology_destroy(topo);               // Cleanup

    return error;                               // EXIT
}
#else
size_t ThreadControl::bind_threads(thread_binding binding)
{
    if (BIND_TO_NONE!=binding) {
        printf("!! Trying to bind but compiled without required library  !!\n"
               "!!            Install hwloc and re-compile               !!\n"
               "!!                          OR                           !!\n"
               "!! Disable binding by setting ENV_VAR: BH_VE_CPU_BIND=0  !!\n"
        );
    }
    return 0;
}
#endif

size_t ThreadControl::bind_threads()
{
    return bind_threads(binding_);
}

}}}

