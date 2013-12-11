/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <bh.h>

#ifndef __BH_VEM_CLUSTER_EXEC_H
#define __BH_VEM_CLUSTER_EXEC_H

//Function pointers to our child.
extern bh_component_iface *mychild;

/* Component interface: init (see bh_component.h) */
bh_error exec_init(const char *component_name);

/* Component interface: shutdown (see bh_component.h) */
bh_error exec_shutdown(void);

/* Component interface: extmethod (see bh_component.h) */
bh_error exec_extmethod(const char *name, bh_opcode opcode);

/* Execute a BhIR where all operands are global arrays
 *
 * @bhir   The BhIR in question
 * @return Error codes
 */
bh_error exec_execute(bh_ir *bhir);

#endif
