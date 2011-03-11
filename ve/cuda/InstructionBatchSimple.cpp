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

#include <iostream>
#include <cphvb.h>
#include "InstructionBatchSimple.hpp"

InstructionBatchSimple::InstructionBatchSimple(Threads threads_, 
                                        DataManager* dataManager_,
                                        KernelGenerator* kernelGenerator_) :
    threads(threads_),
    dataManager(dataManager_),
    kernelGenerator(kernelGenerator_) 
{
#ifdef DEBUG
    std::cout << "[VE CUDA] Created InstructionBatch with " << threads 
              << " threads." << std::endl;
#endif
}


void InstructionBatchSimple::add(cphVBinstruction* inst)
{
#ifdef DEBUG
    std::cout << "[VE CUDA] Adding instruction to batch with dimentions: ";
    for (int i = 0; i < inst->operand[0]->ndim; ++i)
    {
        std::cout << inst->operand[0]->shape[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "[VE CUDA] Adding instruction to batch with stride: ";
    for (int i = 0; i < inst->operand[0]->ndim; ++i)
    {
        std::cout << inst->operand[0]->stride[i] << " ";
    }
    std::cout << std::endl;
#endif
    int nops = cphvb_operands(inst->opcode);
    dataManager->lock(inst->operand, nops, this);
    batch.push_back(inst);
}

void InstructionBatchSimple::execute()
{
    if (batch.begin() != batch.end())
    {
        kernelGenerator->run(threads, batch.begin(), batch.end());
        batch.clear();
    }
}
