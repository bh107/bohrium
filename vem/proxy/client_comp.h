/*
 * client_exec.h
 *
 *  Created on: Feb 3, 2014
 *      Author: d
 */

#ifndef CLIENT_EXEC_H_
#define CLIENT_EXEC_H_

#include <bh.h>

/* Component interface: init (see bh_component.h) */
bh_error client_comp_init(const char *component_name);

/* Component interface: shutdown (see bh_component.h) */
bh_error client_comp_shutdown(void);

/* Component interface: extmethod (see bh_component.h) */
bh_error client_comp_extmethod(const char *name, bh_opcode opcode);

/* Execute a BhIR where all operands are global arrays
 *
 * @bhir   The BhIR in question
 * @return Error codes
 */
bh_error client_comp_execute(bh_ir *bhir);



#endif /* CLIENT_EXEC_H_ */
