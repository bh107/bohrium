#ifndef __KP_ACC_H
#define __KP_ACC_H 1

#include <stdbool.h>
#include "kp.h"

#ifdef __cplusplus
extern "C" {
#endif

// Note: kp_acc is forward declared in kp.h and defined in kp_acc.c

kp_acc* kp_acc_create(size_t id);
int kp_acc_init(kp_acc* acc);
void kp_acc_destroy(kp_acc* acc);

size_t kp_acc_id(kp_acc*);
const char* kp_acc_text(kp_acc*);
size_t kp_acc_bytes_allocated(kp_acc*);

bool kp_acc_allocated(kp_acc* acc, kp_buffer* buffer);
bool kp_acc_pushed(kp_acc* acc, kp_buffer* buffer);

void kp_acc_alloc(kp_acc* acc, kp_buffer* buffer);
void kp_acc_free(kp_acc* acc, kp_buffer* buffer);
void kp_acc_push(kp_acc* acc, kp_buffer* buffer);
void kp_acc_pull(kp_acc* acc, kp_buffer* buffer);

void kp_acc_pprint(kp_acc* acc);

#ifdef __cplusplus
}
#endif

#endif /* __KP_ACC_H */
