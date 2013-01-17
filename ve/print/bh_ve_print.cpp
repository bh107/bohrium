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

#include <iostream>
#include <bh.h>

#ifdef __cplusplus
extern "C" {
#endif

static bh_component* component = NULL;

bh_error bh_ve_print_init(bh_component* _component)
{
    component = _component;
    return CPHVB_SUCCESS;
}

bh_error bh_ve_print_execute(bh_intp instruction_count,
                                   bh_instruction instruction_list[])
{
    std::cout << "# ----------------------------- Recieved batch with " << 
        instruction_count << 
        " instructions --------------------------------------- #" << std::endl;
    for (bh_intp i = 0; i < instruction_count; ++i)
        bh_pprint_instr(instruction_list+i);
    return CPHVB_SUCCESS;
}

bh_error bh_ve_print_shutdown()
{
    return CPHVB_SUCCESS;
}

bh_error bh_ve_print_reg_func(char *fun, 
                                  bh_intp *id)
{
    return CPHVB_SUCCESS;
}

#ifdef __cplusplus
}
#endif
