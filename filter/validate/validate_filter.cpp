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
#include <bh_flow.h>
#include <stdio.h>

bh_error validate_instruction(bh_instruction* instr)
{
    bh_error res = BH_SUCCESS;

    int type_sig = bh_type_sig(instr);
    if (!bh_type_sig_check(type_sig)) {
        printf("validate_filter{ Invalid type signature[%ld];"
                " Bridge check yourself, you created the following invalid instruction:\n", (long)type_sig);
        bh_pprint_instr(instr);
        printf("\n} ");
        res = BH_ERROR;
    }

    return res;
}

void validate_filter(bh_ir *bhir)
{
    bh_ir_map_instr(bhir, &bhir->dag_list[0], &validate_instruction);    
}

