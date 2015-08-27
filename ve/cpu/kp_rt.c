#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(CAPE_WITH_HWLOC)
#include <hwloc.h>
#endif
#if defined(CAPE_WITH_THREADBINDING)
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
inline int omp_get_num_threads() { return 1; }
#endif

#include "kp_rt.h"
#include "kp_vcache.h"
#include "kp_acc.h"
#include "kp_utils.h"

#if defined(CAPE_WITH_HWLOC)
void kp_hw_mic_text(char* text, size_t devid)
{
    strncpy(text, "[COPROC:UNKNOWN]", 16);
/*
    hwloc_topology_t topo;                      // Setup topology
    hwloc_topology_init(&topo);                 
    hwloc_topology_set_flags(topo, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
    hwloc_topology_load(topo);

    const int nobjects = 1;                       // Search these objects
    hwloc_obj_type_t objects[nobjects] = { 
        HWLOC_OBJ_OS_DEVICE 
    };

    string coproc_type;

    for(int i=0; i<nobjects; ++i) {               // For Architecture and CPUModel
        hwloc_obj_t obj = hwloc_get_obj_by_type(topo, objects[i], 0); 
        if (obj) {
            if (coproc_type.empty()) {
                const char *str = hwloc_obj_get_info_by_name(obj, "CoProcType");
                if (str) {
                    coproc_type = string(str);
                }
            }
        }
    }

    hwloc_topology_destroy(topo);               // Cleanup

    stringstream ss;                            // Construct the CPU-text

    ss << "[COPROC:" << (coproc_type.empty() ? "UNKNOWN" : coproc_type) << "]";
    cout << ss.str() << endl;

    return ss.str();
*/
}
#else
void kp_hw_mic_text(char* text, size_t devid)
{
    strncpy(text, "[COPROC:UNKNOWN]", 16);
}
#endif

#if defined(CAPE_WITH_HWLOC)
void kp_set_host_text(char* host_text)
{
    if (host_text) {
        hwloc_topology_t topo;                      // Setup topology
        hwloc_topology_init(&topo);                 
        hwloc_topology_load(topo);

        hwloc_obj_type_t objects[] = {HWLOC_OBJ_MACHINE, HWLOC_OBJ_SOCKET};
        const int nobjects = sizeof(objects)/sizeof(hwloc_obj_type_t);

        strncpy(host_text, "[", 2);                 // Prefix

        bool got_arch = false;                      // Grab architecture
        strncat(host_text, "A=", 3);
        for(int i=0; i<nobjects; ++i) {
            hwloc_obj_t obj = hwloc_get_obj_by_type(topo, objects[i], 0); 
            if (obj) {
                const char *str = hwloc_obj_get_info_by_name(obj, "Architecture"); 
                if (str) {
                    strncat(host_text, str, 50);
                    got_arch = true;
                    break;
                }
            }
        }
        if (!got_arch) {
            strncat(host_text, "UNKNOWN", 8);
        }

        strncat(host_text, ", ", 3);                // Separator

        bool got_model = false;                     // Grab CPUModel
        strncat(host_text, "M=", 3);
        for(int i=0; i<nobjects; ++i) {               
            hwloc_obj_t obj = hwloc_get_obj_by_type(topo, objects[i], 0); 
            if (obj) {
                const char *str = hwloc_obj_get_info_by_name(obj, "CPUModel"); 
                if (str) {
                    strncat(host_text, str, 50);
                    got_model = true;
                    break;
                }
            }
        }
        if (!got_model) {
            strncat(host_text, "UNKNOWN", 7);
        }

        strncat(host_text, "]", 2);                 // Postfix
    }
}
#else
void kp_set_host_text(char* host_text)
{
    if (host_text) {
        strncpy(host_text, "[MODEL:UNKNOWN,ARCH:UNKNOWN]", 30);
    }
}
#endif

kp_rt* kp_rt_create(size_t vcache_size)
{
    kp_rt* rt = malloc(sizeof(*rt));
    if (rt) {
        rt->binding = KP_BIND_TO_NONE;  // No binding until instructed to

        rt->vcache = kp_vcache_create(vcache_size);
        rt->host_text = malloc(sizeof(*(rt->host_text))*200);
        kp_set_host_text(rt->host_text);

        rt->acc = NULL;                 // No accelerators until kp_rt_acc_init(...)
    }
    return rt;
}

void kp_rt_destroy(kp_rt* rt)
{
    if (rt) {
        if (rt->acc) {                  // Free accelerator
            kp_acc_destroy(rt->acc);
        }
        if (rt->host_text) {            // Free host textual identifier
            free(rt->host_text);
        }
        kp_vcache_destroy(rt->vcache);  // De-allocate the victim cache
        free(rt);                       // Free runtime struct
    }
}

bool kp_rt_execute(kp_rt* rt, kp_program* program, kp_symboltable* symbols, kp_block* block, kp_krnl_func func)
{
    //
    // Buffer Management
    //
    // - allocate output buffer(s) on host
    // - allocate output buffer(s) on accelerator
    // - allocate input buffer(s) on accelerator
    // - push input buffer(s) to accelerator (TODO)
    //
    for(size_t i=0; i<block->narray_tacs; ++i) {
        kp_tac* tac = &program->tacs[block->array_tacs[i]];

        if (!((tac->op & KP_ARRAY_OPS)>0)) {
            continue;
        }
        switch(kp_tac_noperands(tac)) {
            case 3:
                if ((rt->acc) && ((symbols->table[tac->in2].layout & (KP_DYNALLOC_LAYOUT))>0)) {
                    kp_acc_alloc(rt->acc, symbols->table[tac->in2].base);
                    if (NULL!=symbols->table[tac->in2].base->data) {
                        kp_acc_push(rt->acc, symbols->table[tac->in2].base);
                    }
                }
            case 2:
                if ((rt->acc) && ((symbols->table[tac->in1].layout & (KP_DYNALLOC_LAYOUT))>0)) {
                    kp_acc_alloc(rt->acc, symbols->table[tac->in1].base);
                    if (NULL!=symbols->table[tac->in1].base->data) {
                        kp_acc_push(rt->acc, symbols->table[tac->in1].base);
                    }
                } 
            case 1:
                if ((symbols->table[tac->out].layout & (KP_DYNALLOC_LAYOUT))>0) {
                    if (!kp_vcache_alloc(rt->vcache, symbols->table[tac->out].base)) {
                        fprintf(stderr, "Unhandled error returned by kpvcache_malloc() "
                                        "called from kp_ve_cpu_execute()\n");
                        return false;
                    }
                    if (rt->acc) {
                        kp_acc_alloc(rt->acc, symbols->table[tac->out].base);
                    }
                }
                break;
        }
    }

    if (func) {                                                     // Execute kernel function
        func(block->buffers, block->operands, &block->iterspace, 0);
    }

    //
    // Buffer Management
    //
    // - free buffer(s) on accelerator
    // - free buffer(s) on host
    // - pull buffer(s) from accelerator to host
    //
    for(size_t i=0; i<block->ntacs; ++i) {
        kp_tac* tac = &program->tacs[block->tacs[i]];
        kp_operand* operand = &symbols->table[tac->out];

        switch(tac->oper) {  

            case KP_SYNC:                                       // Pull buffer from accelerator to host
                if (rt->acc) {
                    kp_acc_pull(rt->acc, operand->base);
                }
                break;

            case KP_DISCARD:                                    // Free buffer on accelerator
                if (rt->acc) {
                    kp_acc_free(rt->acc, operand->base);
                }
                break;

            case KP_FREE:
                if (rt->acc) {                                      // Free buffer on accelerator
                    kp_acc_free(rt->acc, operand->base);            // Note: must be done prior to
                }                                                   //       freeing on host.

                if (!kp_vcache_free(rt->vcache, operand->base)) {  // Free buffer on host
                    fprintf(stderr, "Unhandled error returned by kp_vcache_free(...) "
                                    "called from kp_rt_execute)\n");
                    return false;
                }
                break;

            default:
                break;
        }
    }

    return true;
}

#if defined(CAPE_WITH_THREADBINDING)
int kp_rt_bind_threads(kp_rt* rt, kp_thread_binding binding)
{
    char* env = getenv("GOMP_CPU_AFFINITY");    // Back off if user specifies a
    if (NULL != env) {                          // a different binding scheme
        binding = KP_BIND_TO_NONE;
    }
    env = getenv("KMP_AFFINITY");
    if (NULL != env) {
        binding = KP_BIND_TO_NONE;
    }

    rt->binding = binding;
    if (KP_BIND_TO_NONE == binding) {           // Early exit when not binding
        return 0;                               // EXIT
    }

    hwloc_topology_t topo;
    int obj_count;
    int depth;
    int error = 0;

    hwloc_topology_init(&topo);                 // Setup topology
    hwloc_topology_load(topo);
    if (KP_BIND_TO_PU == binding) {
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
int kp_rt_bind_threads(kp_rt* rt, kp_thread_binding binding)
{
    if (KP_BIND_TO_NONE!=binding) {
        printf("!! Trying to bind but compiled without required library  !!\n"
               "!!            Install hwloc and re-compile               !!\n"
               "!!                          OR                           !!\n"
               "!! Disable binding by setting ENV_VAR: BH_VE_CPU_BIND=0  !!\n"
        );
    }
    return 0;
}
#endif

kp_thread_binding kp_rt_thread_binding(kp_rt* rt)
{
    return rt->binding;
}

size_t kp_rt_vcache_size(kp_rt* rt)
{
    return kp_vcache_capacity(rt->vcache);
}

const char* kp_rt_host_text(kp_rt* rt)
{
    return rt->host_text;
}

void kp_rt_pprint(kp_rt* rt)
{
    if (rt) {
        fprintf(stdout,
                "rt(%p) { binding:%d, vcache_size: %ld, acc: %p, host_text: %s }\n",
                (void*)rt,
                (int)kp_rt_thread_binding(rt),
                kp_rt_vcache_size(rt),
                (void*)rt->acc,
                kp_rt_host_text(rt));
        kp_acc_pprint(rt->acc);
    } else {
        fprintf(stdout, "rt(%p) {}\n", (void*)rt);
    }
}

