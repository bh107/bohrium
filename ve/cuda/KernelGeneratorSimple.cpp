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

#include <queue>
#include <map>
#include <cassert>
#include <cphvb.h>
#include "Configuration.hpp"
#include "KernelGeneratorSimple.hpp"
#include "OffsetMapSimple.hpp"
#include "PTXtype.h"
#include "KernelShapeSimple.hpp"
#include "KernelSimple.hpp"

KernelGeneratorSimple::KernelGeneratorSimple() :
    kernelID(0),
    offsetMap(createOffsetMap()),
    registerBank(new PTXregisterBank()),
    constantBuffer(new StaticStack<PTXconstant>(1024)),
    instructionList(new StaticStack<PTXinstruction>(2048))
{
    translator = new InstructionTranslator(instructionList,registerBank);
    ptxKernel = new PTXkernel(ISA_14,SM_12,registerBank,instructionList);
}

void KernelGeneratorSimple::clear()
{
    elementMap.clear();
    storeMap.clear();
    addressMap.clear();
    offsetMap->clear();
    registerBank->clear();
    constantBuffer->clear();
    instructionList->clear();
    ptxKernel->clear();
    parameters.clear();
}

static PTXregister* dimIndex;
static PTXconstant* zero;
void KernelGeneratorSimple::init(Threads threads)
{
    sprintf(ptxKernel->name,"kernel_%d",kernelID++);
    threadID = registerBank->next(PTX_UINT32);
    PTXregister* tidx = registerBank->next(PTX_UINT32);
    PTXregister* ntidx = registerBank->next(PTX_UINT32);
    PTXregister* ctaidx = registerBank->next(PTX_UINT32);
    PTXregister* skip = registerBank->next(PTX_PRED);
    instructionList->next(PTX_MOV, tidx, &registerBank->tid_x);
    instructionList->next(PTX_MOV, ntidx, &registerBank->ntid_x);
    instructionList->next(PTX_MOV, ctaidx, &registerBank->ctaid_x);
    instructionList->next(PTX_MAD, threadID, ntidx, ctaidx, tidx);
    instructionList->next(PTX_SETP_GE, skip, threadID, 
                             constantBuffer->next(threads));
    instructionList->next(skip,PTX_EXIT);
    dimIndex = registerBank->next(PTX_UINT32);
    zero = constantBuffer->next(0UL);
}

PTXregister* KernelGeneratorSimple::calcOffset(const cphVBarray* array)
{
    cphvb_index dimbound[CPHVB_MAXDIM];
    cphvb_dimbound(array->ndim, array->shape, dimbound);
    PTXregister* offsetReg = registerBank->next(PTX_UINT32);
    instructionList->next(PTX_ADD, offsetReg, zero, zero);
    for (int i = 0; i < array->ndim; ++i)
    {
        if (array->shape[i] == 1 || array->stride[i] == 0) {continue;}
        instructionList->next(PTX_REM, dimIndex, threadID,
                             constantBuffer->next(dimbound[i])); 
        if (i != array->ndim - 1)
        {
            instructionList->next(PTX_DIV, dimIndex, dimIndex, 
                                 constantBuffer->next(dimbound[i+1]));
        }
        instructionList->next(PTX_MAD, offsetReg, dimIndex, 
                             constantBuffer->next(array->stride[i]), 
                             offsetReg);
    }
    offsetMap->insert(array, offsetReg);
    return offsetReg;
}

PTXaddress KernelGeneratorSimple::calcAddress(const cphVBarray* array)
{
    PTXregister* offsetReg = offsetMap->find(array);
    if (offsetReg == NULL)
    {
        offsetReg  = calcOffset(array);
    }
    PTXregister* addressReg;
    long int eSize = cphvb_type_size(array->type);
    switch (sizeof(void*))
    {
    case 4:
        addressReg = registerBank->next(PTX_UINT32);
        instructionList->next(PTX_MAD, addressReg, offsetReg, 
                             constantBuffer->next(eSize),
                             constantBuffer->next(array->cudaPtr));
        break;
    case 8:
        addressReg = registerBank->next(PTX_UINT64);
        instructionList->next(PTX_MAD_WIDE, addressReg, offsetReg, 
                             constantBuffer->next(eSize),
                             constantBuffer->next(array->cudaPtr));
        break;
    default:
        assert (false);
    }
    PTXconstant* off = constantBuffer->next(eSize*array->start);
    addressMap[array] = {addressReg, off};
    return {addressReg, off};
}

PTXregister* KernelGeneratorSimple::loadElement(const cphVBarray* array)
{
    PTXaddress address = calcAddress(array);
    PTXregister* elementReg = registerBank->next(array->type);
    instructionList->next(PTX_LD_GLOBAL, elementReg, address.reg, address.off);
    return elementReg;
}


PTXregister* KernelGeneratorSimple::loadScalar(cphvb_type type,
                                               cphvb_constant value)
{
    PTXkernelParameter* ptxParam = ptxKernel->addParameter(ptxType(type));
    parameters.push_back(KernelParameter(type,value));
    PTXregister* scalarReg = registerBank->next(type);
    instructionList->next(PTX_LD_PARAM, scalarReg, ptxParam);  
    return scalarReg;
}

void KernelGeneratorSimple::addInstruction(const cphVBinstruction* inst)
{
    assert(inst->operand[0] != CPHVB_CONSTANT);
    int nops = cphvb_operands(inst->opcode);
    PTXregister* dest;
    PTXregister* src[CPHVB_MAX_NO_OPERANDS-1];

    // get register for result
    ElementMap::iterator eiter = elementMap.find(inst->operand[0]);
    if (eiter == elementMap.end())
    {
        dest = registerBank->next(inst->operand[0]->type);
        elementMap[inst->operand[0]] = dest;
        storeMap[dest] = inst->operand[0];
    }
    else
    {
        dest = eiter->second;
    }
    
    //get registers for input
    for (int i = 1; i < nops; ++i)
    {
        if (inst->operand[i] != CPHVB_CONSTANT)
        {
            eiter = elementMap.find(inst->operand[i]);
            if (eiter == elementMap.end())
            {
                src[i-1] = loadElement(inst->operand[i]);
                elementMap[inst->operand[i]] = src[i-1];
            }
            else
            {
                src[i-1] = eiter->second;
            }
        }
        else
        {   //Constant
            src[i-1] = loadScalar(inst->const_type[i], inst->constant[i]);
        }
    }
    //generate PTX instruction
    translator->translate(inst->opcode, dest, src);
}

void KernelGeneratorSimple::storeAll()
{
    StoreMap::iterator siter = storeMap.begin();
    AddressMap::iterator aiter;
    PTXaddress address;
    for (;siter != storeMap.end(); ++siter)
    {
        aiter = addressMap.find(siter->second);
        if (aiter != addressMap.end())
        {
            address = aiter->second;
        }
        else
        {
            address = calcAddress(siter->second);
        }
        instructionList->next(PTX_ST_GLOBAL, siter->first, 
                             address.reg, address.off);

    }
}

void KernelGeneratorSimple::run(Threads threads,
                                InstructionIterator first,
                                InstructionIterator last)
{
    init(threads);
    InstructionIterator it;
    for (it = first ;it != last; ++it)
    {
        addInstruction(*it);
    }
    storeAll();
   
    KernelShapeSimple* shape = new KernelShapeSimple(threads);
    KernelSimple* kernel = new KernelSimple(ptxKernel,shape);
   
    kernel->execute(parameters);
    clear();
}
