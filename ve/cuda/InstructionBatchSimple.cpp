/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cphvb.h>
#include "InstructionBatchSimple.hpp"

InstructionBatchSimple::InstructionBatchSimple(Threads threads_, 
                                   DataManager* dataManager_,
                                   KernelGeneratorSimple* kernelGenerator_) :
    threads(threads_),
    dataManager(dataManager_),
    kernelGenerator(kernelGenerator_) {}


void InstructionBatchSimple::add(cphVBInstruction* inst)
{
    int nops = cphvb_operands(inst->opcode);
    dataManager->lock(inst->operand, nops, this);
    batch.push_back(inst);
}

void InstructionBatchSimple::execute()
{
    kernelGenerator->run(this);
}
