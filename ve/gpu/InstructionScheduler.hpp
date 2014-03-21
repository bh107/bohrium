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


#ifndef __INSTRUCTIONSCHEDULER_HPP
#define __INSTRUCTIONSCHEDULER_HPP

#include <map>
#include <set>
#include <bh.h>
#include "InstructionBatch.hpp"
#include "ResourceManager.hpp"

class InstructionScheduler
{
private:
    typedef std::map<bh_base*, BaseArray*> ArrayMap;
    typedef std::map<bh_opcode, bh_extmethod_impl> FunctionMap;
    ResourceManager* resourceManager;
    InstructionBatch* batch;
    ArrayMap arrayMap;
    FunctionMap functionMap;
    std::set<BaseArray*> discardSet;
    void sync(bh_base* base);
    void discard(bh_base* base);
    void executeBatch();
    std::vector<KernelParameter*> getKernelParameters(bh_instruction* inst);
    bh_error ufunc(bh_instruction* inst);
    bh_error reduce(bh_instruction* inst);
    bh_error accumulate(bh_instruction* inst);
    bh_error random(bh_instruction* inst);
    bh_error extmethod(bh_instruction* inst);
public:
    InstructionScheduler(ResourceManager* resourceManager);
    void registerFunction(bh_opcode opcode, bh_extmethod_impl extmothod);
    bh_error schedule(bh_ir* bhir);
};

#endif
