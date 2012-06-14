/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef __INSTRUCTIONSCHEDULER_HPP
#define __INSTRUCTIONSCHEDULER_HPP

#include <map>
#include <set>
#include <cphvb.h>
#include "InstructionBatch.hpp"
#include "ResourceManager.hpp"

class InstructionScheduler
{
private:
    typedef std::map<cphvb_array*, BaseArray*> ArrayMap;
    typedef std::map<cphvb_intp, cphvb_userfunc_impl> FunctionMap;
    ResourceManager* resourceManager;
    InstructionBatch* batch;
    ArrayMap arrayMap;
    FunctionMap functionMap;
    std::set<BaseArray*> discardSet;
    void sync(cphvb_array* base);
    void discard(cphvb_array* base);
    void executeBatch();
    cphvb_error ufunc(cphvb_instruction* inst);
    cphvb_error userdeffunc(cphvb_userfunc* userfunc);
public:
    InstructionScheduler(ResourceManager* resourceManager);
    void registerFunction(cphvb_intp id, cphvb_userfunc_impl userfunc);
    cphvb_error schedule(cphvb_intp instructionCount,
                         cphvb_instruction* instructionList);
};

#endif
