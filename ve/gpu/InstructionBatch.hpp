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

#ifndef __INSTRUCTIONBATCH_HPP
#define __INSTRUCTIONBATCH_HPP

#include <vector>
#include <map>
#include <set>
#include "BaseArray.hpp"

typedef std::map<BaseArray*, cphvb_array*> Output;

class BatchException 
{
private:
    int code;
public:
    BatchException(int code_) : code(code_) {}
};


class InstructionBatch
{
private:
    std::vector<cphvb_instruction*> instructions;
    Output output;
    std::set<BaseArray*> input;
    void accept(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase);
public:
    InstructionBatch(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase);
    void add(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase);
    bool use(BaseArray* baseArray);
    //KernelSource generateKernel();
};

#endif
