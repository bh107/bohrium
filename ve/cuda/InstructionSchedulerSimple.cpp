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
#include "InstructionSchedulerSimple.hpp"

InstructionSchedulerSimple::InstructionSchedulerSimple(
    DataManager* dataManager_) :
    dataManager(dataManager_) {}

void InstructionSchedulerSimple::scedule(cphVBInstruction* inst)
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
        Threads threads = cphvb_nelements(inst->operand[0]->ndim, 
                                          inst->operand[0]->shape);
        BatchTable::iterator biter = batchTable.find(threads);
        if (biter != batchTable.end())
        {
            biter->second->add(inst);
        }
        else
        {
            InstructionBatchSimple* newBatch = 
                new InstructionBatchSimple(dataManager,threads);
            newBatch->add(inst);
            batchTable[threads] = newBatch;                
        }
    }
}


