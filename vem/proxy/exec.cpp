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

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <bh.h>
#include <bh_component.h>

#include "exec.h"

namespace bohrium {
namespace proxy {

//Our self
bh_component myself;

//Function pointers to our child.
bh_component_iface *mychild;

//Known extension methods
static std::map <bh_opcode, bh_extmethod_impl> extmethod_op2impl;

/* Component interface: init (see bh_component.h) */
bh_error exec_init(const char *component_name) {
    bh_error err;

    if ((err = bh_component_init(&myself, component_name)) != BH_SUCCESS)
        return err;

    //For now, we have one child exactly
    if (myself.nchildren != 1) {
        std::cerr << "[CLUSTER-VEM] Unexpected number of children, must be 1" << std::endl;
        return BH_ERROR;
    }

    //Let us initiate the child.
    mychild = &myself.children[0];
    if ((err = mychild->init(mychild->name)) != 0)
        return err;

    return BH_SUCCESS;
}

/* Component interface: shutdown (see bh_component.h) */
bh_error exec_shutdown(void) {
    bh_error err;
    if ((err = mychild->shutdown()) != BH_SUCCESS)
        return err;
    bh_component_destroy(&myself);

    return BH_SUCCESS;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error exec_extmethod(const char *name, bh_opcode opcode) {
    bh_extmethod_impl extmethod;
    bh_error err = bh_component_extmethod(&myself, name, &extmethod);
    if (err != BH_SUCCESS)
        return err;

    if (extmethod_op2impl.find(opcode) != extmethod_op2impl.end()) {
        printf("[PROXY-VEM] Warning, multiple registrations of the same"
                       "extension method '%s' (opcode: %d)\n", name, (int) opcode);
    }
    extmethod_op2impl[opcode] = extmethod;
    return BH_SUCCESS;
}

/* Execute a BhIR where all operands are global arrays
 *
 * @bhir   The BhIR in question
 * @return Error codes
 */
bh_error exec_execute(bh_ir *bhir) {

    return mychild->execute(bhir);
}

/* Returns the initiated self component */
bh_component *exec_get_self_component()
{
    return &myself;
}


}}
