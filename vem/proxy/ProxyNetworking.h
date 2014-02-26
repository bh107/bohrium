/*
 * ProxyNetworking.h
 *
 *  Created on: Feb 4, 2014
 *      Author: d
 */

#ifndef PROXYNETWORKING_H_
#define PROXYNETWORKING_H_

#include <bh.h>

#ifdef __cplusplus
extern "C" {
#endif
//#define PROXY_DEBUG

int Init_Networking(uint16_t port);
int Shutdown_Networking();

bh_error nw_init(const char *component_name);
bh_error nw_shutdown();
bh_error nw_execute(bh_ir *bhir);
bh_error nw_extmethod(const char *name, bh_opcode opcode);

#ifdef __cplusplus
}
#endif

#endif /* PROXYNETWORKING_H_ */
