#ifndef __CPHVB_PRIVATE_H
#define __CPHVB_PRIVATE_H
#include "cphvb.h"

/* Setup the pointers in cphvb_instruktion struct for seri
 *
 * @inst   Will be initialized with pointers
 * @seri   Start of the data area that contains the instruction
 * @return Pointer to after stride[][].
 */
char* _setup_pointers(cphvb_instruction* inst, 
                      char* seri);

#endif
