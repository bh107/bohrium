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

#ifndef __KERNELSIMPLE_HPP
#define __KERNELSIMPLE_HPP

#include <queue>
#include <map>
#include "OffsetMap.hpp"
#include "PTXKernelBody.hpp"
#include "PTXKernel.hpp"
#include "Kernel.hpp"
#include "KernelParameter.hpp"

typedef std::map<cphVBArray*, Register*> ElementMap;
typedef std::map<Register*, cphVBArray*> StoreMap;

class KernelGeneratorSimple
{
private:
    ElementMap elementMap;
    StoreMap storeMap;
    PTXregisterBank* registerBank;
    PTXconstantBuffer* constantBuffer;
    OffsetMap* offsetMap;
    PTXkernel* ptxKernel;
    PTXkernelBody* instructionList;
    PTXregister* threadID;
    ParameterList parameters;
public:
    KernelGeneratorSimple();
    void addInstruction(cphVBInstruction* inst);
};

#endif
