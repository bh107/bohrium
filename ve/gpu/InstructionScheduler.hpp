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
#include "cphVBinstruction.hpp"
#include "InstructionBatch.hpp"
#include "KernelGenerator.hpp"
#include "DataManager.hpp"
#include "ReduceTB.hpp"

typedef std::map<unsigned long workItems, InstructionBatch*> BatchTable;

class InstructionScheduler
{
private:
    DataManager* dataManager;
    KernelGenerator* kernelGenerator;
    BatchTable batchTable;
    RandomNumberGenerator* randomNumberGenerator;
protected:
    void schedule(cphVBinstruction* inst);
public:
    InstructionScheduler(DataManager* dataManager_,
                         KernelGenerator* kernelGenerator_);
    void schedule(cphvb_intp instructionCount,
                  cphVBinstruction* instructionList);
    void flushAll();
};

#endif
