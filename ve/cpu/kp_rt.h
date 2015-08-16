#ifndef __KP_RT_H
#define __KP_RT_H 1

#include <stdbool.h>
#include "kp.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  Query through hwloc for a textual machine identifier.
 *
 *  Caller is responsible for allocating and deallocating pointer.
 *  Pointer should have room for atleast 200 chars.
 */
void kp_set_host_text(char* host_text);

/**
 *  Initialize the CAPE C-Runtime
 *
 *  Returns pointer to the initialized runtime.
 */
kp_rt* kp_rt_create(size_t vcache_size);

/**
 *  Shut down the CAPE C-Runtime
 */
void kp_rt_destroy(kp_rt* rt);

/**
 *  Execute the given block using the provided runtime.
 */
bool kp_rt_execute(kp_rt* rt, kp_program* program, kp_symboltable* symbols, kp_block* block, kp_krnl_func func);

/**
 *  Bind threads on host PUs.
 */
int kp_rt_bind_threads(kp_rt* rt, kp_thread_binding binding);

/**
 *  Get the thread-binding performed with kp_rt_bind_threads.
 */
kp_thread_binding kp_rt_thread_binding(kp_rt* rt);

/**
 *  Get max # of buffers in the victim cache.
 */
size_t kp_rt_vcache_size(kp_rt* rt);

/**
 *  Get the textual host representation.
 */
const char* kp_rt_host_text(kp_rt* rt);

/**
 *  Pretty print the runtime.
 */
void kp_rt_pprint(kp_rt* rt);

#ifdef __cplusplus
}
#endif

#endif /* __KP_RT_H */
