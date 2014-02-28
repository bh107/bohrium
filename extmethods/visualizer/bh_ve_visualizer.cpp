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
#include <stdio.h>
#include <bh.h>
#include <vector>
#include <cassert>
#include "bh_ve_visualizer.h"
#include "bh_visualizer.h"

static bh_component myself; // Myself

//Function pointers to our child.
static bh_component_iface *child;


bh_error bh_ve_visualizer_init(const char* name)
{
    bh_error err;
    if((err = bh_component_init(&myself, name)) != BH_SUCCESS)
        return err;

    //For now, we have one child exactly
    if(myself.nchildren != 1)
    {
        fprintf(stderr, "[VE-VISUALIZER] Unexpected number of children, must be 1");
        return BH_ERROR;
    }

    //Let us initiate the child.
    child = &myself.children[0];
    if((err = child->init(child->name)) != 0)
        return err;

    return BH_SUCCESS;
}

static std::vector<bh_instruction> batch;
static bh_opcode visualizer_opcode;

static bh_error inspect(bh_instruction *instr)
{
    batch.push_back(*instr);
    return BH_SUCCESS;
}

bh_error bh_ve_visualizer_execute(bh_ir* bhir)
{
    //Convert BhIR to the instruction list 'batch'
    bh_ir_map_instr(bhir, &bhir->dag_list[0], &inspect);

    bh_intp i=0, start=0;
    for(i=0; i<(int)batch.size(); ++i)
    {
        //Lets find the next visualizer opcode
        while(i<(int)batch.size() && batch[i].opcode != visualizer_opcode)
            ++i;

        if(i >= (int)batch.size())
            break;//We did not find any visualize instruction

        assert(batch[i].opcode == visualizer_opcode);

        bh_intp size = i-start;
        if(size > 0)//Execute instruction up until the visualize
        {
            bh_ir new_bhir;
            bh_error e = bh_ir_create(&new_bhir, size, &batch[start]);
            if(e != BH_SUCCESS)
                return e;
            e = child->execute(&new_bhir);
            if(e != BH_SUCCESS)
                return e;
            bh_ir_destroy(&new_bhir);
        }
        //Execute the visualize instruction after a SYNC
        bh_ir new_bhir;
        bh_instruction instr_list[2];
        instr_list[0].opcode = BH_SYNC;
        instr_list[0].operand[0] = batch[i].operand[0];
        instr_list[1].opcode = BH_SYNC;
        instr_list[1].operand[0] = batch[i].operand[1];
        bh_error e = bh_ir_create(&new_bhir, 2, instr_list);
        if(e != BH_SUCCESS)
            return e;
        e = child->execute(&new_bhir);
        if(e != BH_SUCCESS)
            return e;
        bh_ir_destroy(&new_bhir);
        start = i+1;

        //Do the visualization
        if((e = bh_visualizer(&batch[i], NULL)) != BH_SUCCESS)
            return e;
    }
    //Execute final batch
    bh_intp size = i-start;
    if(size > 0)
    {
        bh_ir new_bhir;
        bh_error e = bh_ir_create(&new_bhir, size, &batch[start]);
        if(e != BH_SUCCESS)
            return e;
        e = child->execute(&new_bhir);
        if(e != BH_SUCCESS)
            return e;
        bh_ir_destroy(&new_bhir);
    }
    batch.clear();
    return BH_SUCCESS;
}

bh_error bh_ve_visualizer_shutdown(void)
{
    bh_error err = child->shutdown();
    bh_component_destroy(&myself);
    return err;
}

bh_error bh_ve_visualizer_extmethod(const char *name, bh_opcode opcode)
{
    if(strcmp(name, "visualizer") == 0)
    {
        printf("The Visualizer-VE will handle the visualizer extmethod\n");
        visualizer_opcode = opcode;
        return BH_SUCCESS;
    }
    return child->extmethod(name, opcode);
}
