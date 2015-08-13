#ifndef __KP_UTILS_H
#define __KP_UTILS_H 1
#include <stdint.h>
#include <stddef.h>

#include "kp.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t kp_etype_nbytes(KP_ETYPE etype);
size_t kp_buffer_nbytes(const kp_buffer* buffer);
size_t kp_tac_noperands(const kp_tac* tac);

#ifdef __cplusplus
}
#endif
#endif
