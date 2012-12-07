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
        case CPHVB_USERFUNC:
//                userdeffunc(inst->userfunc);
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
    cphvb_array* out = cphvb_base_array(inst->operand[0]);
    auto omsub = lastModifiedBy.find(out);
    if (omsub != lastModifiedBy.end())
    {
        /* If another subGraph is already writing to the same base array
         * We just create a new subGraph, and make it dependent on it
         * TODO: Pruning and so on 
         */
        DepSupGraph* sub = new DepSubGraph(inst, std::list<DepSubGraph*>(1,omsub));
        subGraphs.push_back(sub);
        lastModifiedBy[out] = sub;
        return;
    }

    cphvb_intp nops = cphvb_operands(inst->opcode);
    std::list<DepSubGraph*> dependsOn;
    for (int i = 1; i < nops; ++i)
    {
        if (!cphvb_is_constant(inst->operand[i]))
        {
            cphvb_array* in =  cphvb_base_array(inst->operand[i]);
            DepSubGraph* imsub = lastModifiedBy.find(in);
            if (imsub != lastModifiedBy.end())
            {
                dependsOn.push_back(imsub);
            }
        }
    }
    switch (dependsOn.size())
    {
    case 0:
        DepSubGraph* sub = new DepSubGraph(inst);
        subGraphs.push_back(sub);
        lastModifiedBy[out] = sub;        
        break;
    case 1:
        dependsOn.front()->add(inst);
        break;
    default:
        try 
        {
            DepSubGraph* sub = DepSubGraph::merge(inst,dependsOn);
            subGraphs.push_back(sub);
            lastModifiedBy[out] = sub;
            for (DepSubGraph* &dsg: dependsOn)
            {
                for (chpvb_array* &ba: dsg->getModified())
                {
                    if (lastModifiedBy[ba] == dsg)
                        lastModifiedBy[ba] = sub;
                }
                subGraphs.remove(dsg);
                delete dsg;
            } 
            catch (DepSubGraphException  e) 
            {
                DepSubGraph* sub = new DepSubGraph(inst,dependsOn);
                subGraphs.push_back(sub);
                lastModifiedBy[out] = sub;
            }
        }
    }
}

