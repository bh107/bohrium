/*
 * ArrayManagement.h
 *
 *  Created on: Feb 6, 2014
 *      Author: d
 */

#ifndef ARRAYMANAGEMENT_H_
#define ARRAYMANAGEMENT_H_

#include <bh.h>

#include "Server_Error.h"
#include "BasicPacket.h"

typedef struct
{
	// the id of the array.
	bh_intp id;

	// the array base
	bh_base array;
} array_header;


// array functions

void ArrayMan_init();

void ArrayMan_shutdown();

bh_error ArrayMan_reset_msg();
void ArrayMan_add_to_payload(bh_intp size, void * data);
bh_error ArrayMan_send_payload(packet_protocol ptc, int filedes);

bh_error ArrayMan_client_bh_ir_package(bh_ir * bhir, int filedes);



#endif /* ARRAYMANAGEMENT_H_ */
