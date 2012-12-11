/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include "DepGraph.hpp"

DepGraph::DepGraph(cphvb_intp instruction_count,
                   cphvb_instruction instruction_list[])
{
    
    for (cphvb_intp i = 0; i < instruction_count; ++i)
    {
        cphvb_instruction* inst = instruction_list++;
        switch (inst->opcode)
        {
        case CPHVB_SYNC:
//                sync(inst->operand[0]);
            break;
        case CPHVB_DISCARD:
            if (inst->operand[0]->base == NULL)
//                    discard(inst->operand[0]);
                break;
        case CPHVB_FREE:
//                free(inst->operand[0]);
            break;                
        case CPHVB_NONE:
            break;
        default:
            ufunc(inst);
            break;
        }
    }
}


void DepGraph::ufunc(cphvb_instruction* inst)
{
    cphvb_intp nin, nout;
    cphvb_array** operand;
    /* Handle bothe ufunc and userdefined functions*/
    if (inst->opcode == CPHVB_USERFUNC)
    {
        cphvb_userfunc* userfunc = inst->userfunc;
        nout = userfunc->nout;
        nin = userfunc->nin;
        operand = userfunc->operand;
    } else {
        nout = 1;
        nin = cphvb_operands(inst->opcode) - 1;
        operand = inst->operand;
    }

    std::list<DepSubGraph*> dependsOn;
    /* Handle dependency on output parameters*/ 
    for (int i = 0; i < nout; ++i)
    {
        /* If another subGraph is already writing to the same base array
         * We just create a new subGraph, and make the new subGraph dependent
         * on other subGraphs with overlapping output
         * TODO: Pruning and so on 
         */
        cphvb_array* out = cphvb_base_array(operand[i]);
        auto omsub = lastModifiedBy.find(out);
        if (omsub != lastModifiedBy.end())
        {
            /* There is depencies on the output */
            dependsOn.push_back(omsub->second);
        }
    }
    if (dependsOn.size() > 0) // Is there dependencies on the output
    {
        // Make a new subgraph
        DepSubGraph* sub = new DepSubGraph(inst, dependsOn);
        subGraphs.push_back(sub);
        for (int i = 0; i < nout; ++i)
        {
            cphvb_array* out = cphvb_base_array(operand[i]);
            lastModifiedBy[out] = sub;
        }
        return;
    }

    /* No dependencies on the output */
    /* Handle dependency on intput parameters*/ 
    for (int i = nout; i < nout+nin; ++i)
    {
        if (!cphvb_is_constant(inst->operand[i]))
        {
            cphvb_array* in =  cphvb_base_array(operand[i]);
            auto imsub = lastModifiedBy.find(in);
            if (imsub != lastModifiedBy.end())
            {
                dependsOn.push_back(imsub->second);
            }
        }
    }
    switch (dependsOn.size())
    {
    case 0: 
    {
        DepSubGraph* sub = new DepSubGraph(inst);
        subGraphs.push_back(sub);
        for (int i = 0; i < nout; ++i)
        {
            cphvb_array* out = cphvb_base_array(operand[i]);
            lastModifiedBy[out] = sub;
        }
        break;
    }
    case 1:
    {
        dependsOn.front()->add(inst);
        break;
    }
    default:
        try 
        {
            DepSubGraph* sub = DepSubGraph::merge(inst,dependsOn);
            subGraphs.push_back(sub);
            for (int i = 0; i < nout; ++i)
            {
                cphvb_array* out = cphvb_base_array(operand[i]);
                lastModifiedBy[out] = sub;
            }
            for (DepSubGraph* &dsg: dependsOn)
            {
                for (cphvb_array* ba: dsg->getModified())
                {
                    if (lastModifiedBy[ba] == dsg)
                        lastModifiedBy[ba] = sub;
                }
                subGraphs.remove(dsg);
                delete dsg;
            }
        }
        catch (DepSubGraphException  e) 
        {
            DepSubGraph* sub = new DepSubGraph(inst,dependsOn);
            subGraphs.push_back(sub);
            for (int i = 0; i < nout; ++i)
            {
                cphvb_array* out = cphvb_base_array(operand[i]);
                lastModifiedBy[out] = sub;
            }
        }
    }
}

