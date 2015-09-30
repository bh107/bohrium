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
#include <bh_instruction.h>
#include "bh_vem_visualizer.h"

using namespace std;

//Function pointers to our child.
static bh_component_iface *child;

//Our self
static bh_component myself;

//The visualizer opcode and implementation
static int visualizer_opcode = -1;
bh_extmethod_impl visualizer_impl;

/* Component interface: init (see bh_component.h) */
bh_error bh_vem_visualizer_init(const char* name)
{
    bh_error err;

    if((err = bh_component_init(&myself, name)) != BH_SUCCESS)
        return err;

    //For now, we have one child exactly
    if(myself.nchildren != 1)
    {
        std::cerr << "[VISUALIZER-VEM] Unexpected number of children, must be 1" << std::endl;
        return BH_ERROR;
    }

    //Let us initiate the child.
    child = &myself.children[0];
    if((err = child->init(child->name)) != 0)
        return err;

    return BH_SUCCESS;
}

/* Component interface: shutdown (see bh_component.h) */
bh_error bh_vem_visualizer_shutdown(void)
{
    bh_error err = child->shutdown();
    bh_component_destroy(&myself);
    return err;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error bh_vem_visualizer_extmethod(const char *name, bh_opcode opcode)
{
    string name_str(name);
    cout << "bh_vem_visualizer_extmethod: " << name << endl;

    if(name_str != "visualizer")
        return child->extmethod(name, opcode);


    bh_error err = bh_component_extmethod(&myself, "visualizer", &visualizer_impl);
    if (err != BH_SUCCESS) {
        cerr << "[VISUALIZER-VEM] Cannot find the visualizer extension method!" << endl;
        return err;
    }
    visualizer_opcode = opcode;
    cout << "reg ext method visualizer with opcode: " << opcode << endl;
    return BH_SUCCESS;
}

//Checks if the visualizer opcode exists in the bhir
static bool visualizer_in_bhir(const bh_ir *bhir)
{
    for(const bh_instruction &instr: bhir->instr_list)
    {
        if(instr.opcode == visualizer_opcode)
            return true;
    }
    return false;
}

/* Component interface: execute (see bh_component.h) */
bh_error bh_vem_visualizer_execute(bh_ir* bhir)
{
    if(not visualizer_in_bhir(bhir))
        return child->execute(bhir);

    size_t exec_count = 0;//Count of already executed instruction
    for(size_t i=0; i<bhir->instr_list.size(); ++i)
    {
        if(bhir->instr_list[i].opcode == visualizer_opcode)
        {
            bh_instruction sync;
            sync.opcode = BH_SYNC;
            sync.operand[0] = bhir->instr_list[i].operand[0];

            if(exec_count < i)//Let's execute the instructions between 'exec_count' and 'i' with an appended SYNC
            {
                bh_ir b;
                b.instr_list.insert(b.instr_list.end(), bhir->instr_list.begin()+exec_count, bhir->instr_list.begin()+i);
                b.instr_list.push_back(sync);
                bh_error ret = child->execute(&b);
                if(ret != BH_SUCCESS)
                    return ret;
                exec_count += i - exec_count;
            }
            else
            {
                bh_ir b;
                b.instr_list.push_back(sync);
                bh_error ret = child->execute(&b);
                if(ret != BH_SUCCESS)
                    return ret;
            }
            //Now let's visualize
            visualizer_impl(&bhir->instr_list[i], NULL);
            exec_count = i;
        }
    }
    bh_ir b;
    b.instr_list.insert(b.instr_list.end(), bhir->instr_list.begin()+exec_count, bhir->instr_list.end());
    return child->execute(&b);
}
