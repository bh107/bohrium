#ifndef __KP_UTILS_H
#define __KP_UTILS_H 1
#include <stdint.h>
#include <stddef.h>

#include <kp_tac.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t kp_etype_nbytes(KP_ETYPE etype);
size_t kp_buffer_nbytes(const kp_buffer* buffer);

#ifdef __cplusplus
}
#endif
#endif
