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

#ifndef __INSTRUCTIONBATCHSIMPLE_HPP
#define __INSTRUCTIONBATCHSIMPLE_HPP

typedef long int Threads;

#include <vector>
#include "InstructionBatch.hpp"
#include "KernelGeneratorSimple.hpp"
#include "DataManager.hpp"

class KernelGeneratorSimple;

class InstructionBatchSimple : public InstructionBatch 
{
    friend class KernelGeneratorSimple;
private:
    Threads threads;
    DataManager* dataManager;
    KernelGeneratorSimple* kernelGenerator;
    std::vector<cphVBInstruction*> batch;
public:
    InstructionBatchSimple(Threads threads,
                           DataManager* dataManager,
                           KernelGeneratorSimple* kernelGenerator);
    void add(cphVBInstruction* inst);
    void execute();
};

#endif
