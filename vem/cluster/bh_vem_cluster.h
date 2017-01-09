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

#ifndef __BH_VEM_CLUSTER_H
#define __BH_VEM_CLUSTER_H


#ifdef __cplusplus
extern "C" {
#endif

/* Component interface: init (see bh_component.h) */
DLLEXPORT bh_error bh_vem_cluster_init(const char* name);

/* Component interface: shutdown (see bh_component.h) */
DLLEXPORT bh_error bh_vem_cluster_shutdown(void);

/* Component interface: extmethod (see bh_component.h) */
DLLEXPORT bh_error bh_vem_cluster_execute(bh_ir* bhir);

/* Component interface: extmethod (see bh_component.h) */
DLLEXPORT bh_error bh_vem_cluster_extmethod(const char *name, bh_opcode opcode);

#ifdef __cplusplus
}
#endif

#endif
