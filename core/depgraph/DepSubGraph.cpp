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

#include "DepSubGraph.hpp"
#include <utility>

DepSubGraph::DepSubGraph(bh_instruction* inst)
{
    instructions.push_back(inst);
    bh_array* obase = bh_base_array(inst->operand[0]); 
    modificationMap.insert(std::make_pair(obase,inst));
    arrayAtributes.insert(std::make_pair(obase,atributes(false,false,false)));
}

DepSubGraph::DepSubGraph(bh_instruction* inst, std::list<DepSubGraph*> _dependsOn)
{
    instructions.push_back(inst); 
    bh_array* obase = bh_base_array(inst->operand[0]); 
    modificationMap.insert(std::make_pair(obase,inst));
    arrayAtributes.insert(std::make_pair(obase,atributes(false,false,false)));
    dependsOn = _dependsOn;
}

void DepSubGraph::add(bh_instruction* inst)
{
    instructions.push_back(inst); 
    bh_array* obase = bh_base_array(inst->operand[0]); 
    modificationMap.insert(std::make_pair(obase,inst));
    arrayAtributes.insert(std::make_pair(obase,atributes(false,false,false)));
}

std::set<bh_array*> DepSubGraph::getModified()
{
    std::set<bh_array*> res;
    for (auto &m: modificationMap)
        res.insert(m.first);
    return res;
}

bool DepSubGraph::depends(DepSubGraph* subGraph)
{
    for (DepSubGraph* dsg: dependsOn)
        if (subGraph == dsg)
            return true;
    return false;
}

DepSubGraph* DepSubGraph::merge(bh_instruction* inst, 
                                std::list<DepSubGraph*> subGraphs)
{
    for (DepSubGraph* sg1: subGraphs)
        for (DepSubGraph* sg2: subGraphs)
            if (sg1 != sg2 && sg1->depends(sg2))
                throw DepSubGraphException(0);
    // Do the merging
    DepSubGraph* res = new DepSubGraph(inst);
    for (DepSubGraph* sg: subGraphs)
    {
        for (bh_instruction* i: sg->instructions)
            res->instructions.push_front(i);
        for (auto &mp: sg->modificationMap)
            res->modificationMap.insert(mp);
        for (DepSubGraph* dep: sg->dependsOn)
            res->dependsOn.push_back(dep);
        for (auto &aa: sg->arrayAtributes)
            res->arrayAtributes.insert(aa);
    }
    return res;
}

