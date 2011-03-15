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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __KERNELGENERATORSIMPLE_HPP
#define __KERNELGENERATORSIMPLE_HPP

#include <map>
#include <StaticContainer.hpp>
#include "KernelGenerator.hpp"
#include "PTXregister.hpp"
#include "PTXconstant.hpp"
#include "PTXregisterBank.hpp"
#include "OffsetMap.hpp"
#include "PTXinstruction.hpp"
#include "PTXkernel.hpp"
#include "Kernel.hpp"
#include "KernelParameter.hpp"
#include "InstructionTranslator.hpp"
#include "cphVBinstruction.hpp"

struct PTXaddress
{
    PTXregister* reg;
    PTXconstant* off;
};

typedef std::map<const cphVBarray*, PTXregister*> ElementMap;
typedef std::map<const cphVBarray*, PTXaddress> AddressMap;
typedef std::map<PTXregister*, const cphVBarray*> StoreMap;

class InstructionBatchSimple;

class KernelGeneratorSimple : public KernelGenerator
{
private:
    int kernelID;
    ElementMap elementMap;
    StoreMap storeMap;
    AddressMap addressMap;
    OffsetMap* offsetMap;
    PTXregisterBank* registerBank;
    StaticContainer<PTXconstant>* constantBuffer;
    StaticContainer<PTXinstruction>* instructionList;
    InstructionTranslator* translator;
    PTXkernel* ptxKernel;
    PTXregister* threadID;
    ParameterList parameters;
    void init(Threads threads);
    void clear();
    void addInstruction(const cphVBinstruction* inst);
    PTXregister* calcOffset(const cphVBarray* array);
    PTXaddress   calcAddress(const cphVBarray* array);
    PTXregister* loadElement(const cphVBarray* array);
    PTXregister* loadScalar(cphvb_type type,
                            cphvb_constant value);
    void storeAll();
public:
    KernelGeneratorSimple();
    void run(Threads threads,
             InstructionIterator first,
             InstructionIterator last);
};

#endif
