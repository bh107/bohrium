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
#include "InstructionSchedulerSimple.hpp"

InstructionSchedulerSimple::InstructionSchedulerSimple(
    DataManager* dataManager_,
    KernelGenerator* kernelGenerator_) :
    dataManager(dataManager_),
    kernelGenerator(kernelGenerator_){}

void InstructionSchedulerSimple::schedule(cphVBinstruction* inst)
{
    switch (inst->opcode)
    {
    case CPHVB_RELEASE:
        dataManager->release(inst->operand[0]);
        break;
    case CPHVB_SYNC:
        dataManager->sync(inst->operand[0]);
        break;
    case CPHVB_DISCARD:
        dataManager->discard(inst->operand[0]);
        break;
    default:
#ifdef DEBUG
    std::cout << "[VE CUDA] InstructionSchedulerSimple::schedule(" <<  
        cphvb_opcode_text(inst->opcode) << " ," << 
        inst->operand[0]  << " ," << 
        inst->operand[1]  << " ," << 
        inst->operand[2]  << ")" << std::endl;
#endif
        Threads threads = cphvb_nelements(inst->operand[0]->ndim, 
                                          inst->operand[0]->shape);
        if (threads == 0)
        {
            return;
        }
        BatchTable::iterator biter = batchTable.find(threads);
        if (biter != batchTable.end())
        {
            biter->second->add(inst);
        }
        else
        {
            InstructionBatchSimple* newBatch = 
                new InstructionBatchSimple(threads, dataManager, 
                                           kernelGenerator);
            newBatch->add(inst);
            batchTable[threads] = newBatch;                
        }
    }
}

void InstructionSchedulerSimple::schedule(cphvb_intp instructionCount,
                                         cphVBinstruction* instructionList)
{

    for (cphvb_intp i = 0; i < instructionCount; ++i)
    {
        schedule(instructionList++);
    }
}

void InstructionSchedulerSimple::flush()
{
    BatchTable::iterator iter = batchTable.begin();
    for (; iter !=  batchTable.end(); ++iter)
    {
        iter->second->execute();
    }
}


