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

#include <iostream>
#include <cphvb.h>
#include "InstructionScheduler.hpp"

InstructionScheduler::InstructionScheduler(DataManager* dataManager_,
                                           KernelGenerator* kernelGenerator_) :
    dataManager(dataManager_),
    kernelGenerator(kernelGenerator_),
    randomNumberGenerator(0)
{}

inline void InstructionScheduler::schedule(cphvb_instruction* inst)
{
#ifdef DEBUG
    std::cout << "[VE GPU] InstructionScheduler::schedule(" << 
        *inst << ")" << std::endl;
#endif
    switch (inst->opcode)
    {
    case CPHVB_NONE:
        break;
    case CPHVB_RELEASE:
        dataManager->release(inst->operand[0]);
        break;
    case CPHVB_SYNC:
        dataManager->sync(inst->operand[0]);
        break;
    case CPHVB_DISCARD:
        dataManager->discard(inst->operand[0]);
        break;
    case CPHVB_RANDOM:
        if(randomNumberGenerator == 0)
        {
            randomNumberGenerator = createRandomNumberGenerator();
        }
        dataManager->lock(inst->operand,1,NULL);
        randomNumberGenerator->fill(inst->operand[0]);
        break;
    default:
        if (inst->opcode & CPHVB_REDUCE)
        {
            //TODO: Implement handleing reduce operations
        }
        else
        {
            unsigned long workItems = cphvb_nelements(inst->operand[0]->ndim, 
						      inst->operand[0]->shape);
            if (workItems == 0)
            {
                return;
            }
            BatchTable::iterator biter = batchTable.find(workItems);
            if (biter != batchTable.end())
            {
                biter->second->add(inst);
            }
            else
            {
                InstructionBatch* newBatch = 
		  new InstructionBatch(workItems, dataManager, 
				       kernelGenerator);
                newBatch->add(inst);
                batchTable[workItems] = newBatch;                
            }
        }
    }
}

void InstructionScheduler::flushAll()
{
    dataManager->flushAll();
}

void InstructionScheduler::schedule(cphvb_intp instructionCount,
                                    cphvb_instruction* instructionList)
{
#ifdef DEBUG
    std::cout << "[VE CUDA] InstructionScheduler: recieved batch with " << 
        instructionCount << " instructions." << std::endl;
#endif
    for (cphvb_intp i = 0; i < instructionCount; ++i)
    {
        schedule(instructionList++);
    }
    
    /* End of batch cleanup */
    dataManager->batchEnd();
}
