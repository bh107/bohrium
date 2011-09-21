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

#include "Configuration.hpp"
#include "DataManagerSimple.hpp"
#include "KernelShapeSimple.hpp"
#include "DeviceManagerSimple.hpp"
#include "KernelSimple.hpp"
#include "InstructionBatchSimple.hpp"
#include "MemoryManagerSimple.hpp"
#include "InstructionSchedulerSimple.hpp"
#include "OffsetMapSimple.hpp"
#include "KernelGeneratorSimple.hpp"
#include "RandomNumberGeneratorHybridTaus.hpp"

DeviceManager* createDeviceManager()
{
    return new DeviceManagerSimple();
}

MemoryManager* createMemoryManager()
{
    return new MemoryManagerSimple();
}

DataManager* createDataManager(MemoryManager* memoryManager)
{
    return new DataManagerSimple(memoryManager);
}

InstructionScheduler* createInstructionScheduler(
    DataManager* dataManager,
    KernelGenerator* kernelGenerator)
{
    return new InstructionSchedulerSimple(dataManager, kernelGenerator);
}


InstructionBatch* createInstructionBatch(Threads threads,
                                         DataManager* dataManager,
                                         KernelGenerator* kernelGenerator)
{
    return new InstructionBatchSimple(threads, dataManager, kernelGenerator);
}

KernelGenerator* createKernelGenerator()
{
    return new KernelGeneratorSimple();
}

OffsetMap* createOffsetMap()
{
    return new OffsetMapSimple();
}

RandomNumberGenerator* createRandomNumberGenerator()
{
    return new RandomNumberGeneratorHybridTaus();
}
