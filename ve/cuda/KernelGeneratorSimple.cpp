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
    constantBuffer(new PTXconstantBuffer()),
    instructionList(new PTXinstructionList())
{
    translator = new InstructionTranslator(instructionList,registerBank);
    ptxKernel = new PTXkernel(ISA_14,SM_12,registerBank,instructionList);
}

void KernelGeneratorSimple::init(Threads threads)
{
    sprintf(ptxKernel->name,"kernel_%d",kernelID++);
    threadID = registerBank->newRegister(PTX_UINT32);
    PTXregister* tidx = registerBank->newRegister(PTX_UINT32);
    PTXregister* ntidx = registerBank->newRegister(PTX_UINT32);
    PTXregister* ctaidx = registerBank->newRegister(PTX_UINT32);
    PTXregister* skip = registerBank->newRegister(PTX_PRED);
    instructionList->add(PTX_MOV, tidx, &registerBank->tid_x);
    instructionList->add(PTX_MOV, ntidx, &registerBank->ntid_x);
    instructionList->add(PTX_MOV, ctaidx, &registerBank->ctaid_x);
    instructionList->add(PTX_MAD, threadID, ntidx, ctaidx, tidx);
    instructionList->add(PTX_SETP_GE, skip, threadID, 
                         constantBuffer->newConstant(PTX_UINT,threads));
    instructionList->add(skip,PTX_EXIT);
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


PTXregister* KernelGeneratorSimple::calcOffset(const cphVBarray* array)
{
    PTXregister* offsetReg = registerBank->newRegister(PTX_UINT32);
    cphvb_intp dim = array->ndim -1;
    cphvb_index dimBound = array->shape[dim];
    PTXregister* dimIndex = registerBank->newRegister(PTX_UINT32);
    PTXconstant* nextBound = constantBuffer->newConstant(PTX_UINT,dimBound);
    PTXconstant* prevBound;
    instructionList->add(PTX_REM, dimIndex, threadID, nextBound);
    instructionList->add(PTX_MUL, offsetReg, dimIndex,  
                   constantBuffer->newConstant(PTX_UINT,array->stride[dim]));
    for (--dim; dim >= 0; --dim)
    {
        if (array->stride[dim] > 0)
        {
            prevBound = nextBound;
            if (dim) // All but last dim
            {
                nextBound = constantBuffer->newConstant(PTX_UINT, dimBound * 
                                                        array->shape[dim]);
                instructionList->add(PTX_REM, dimIndex, threadID, nextBound);
                instructionList->add(PTX_DIV, dimIndex, dimIndex, prevBound);
            }
            else // Last dim
            {
                instructionList->add(PTX_DIV, dimIndex, threadID, prevBound);
            }
            instructionList->add(PTX_MAD, offsetReg, dimIndex, 
                     constantBuffer->newConstant(PTX_UINT,array->stride[dim]),
                     offsetReg);
            dimBound *= array->shape[dim];
        }
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
        addressReg = registerBank->newRegister(PTX_UINT32);
        instructionList->add(PTX_MAD, addressReg, offsetReg, 
                   constantBuffer->newConstant(PTX_UINT,eSize),
                   constantBuffer->newConstant(PTX_ADDRESS,array->cudaPtr));
        break;
    case 8:
        addressReg = registerBank->newRegister(PTX_UINT64);
        instructionList->add(PTX_MAD_WIDE, addressReg, offsetReg, 
                   constantBuffer->newConstant(PTX_UINT,eSize),
                   constantBuffer->newConstant(PTX_ADDRESS,array->cudaPtr));
        break;
    default:
        assert (false);
    }
    PTXconstant* off = constantBuffer->newConstant(PTX_UINT,eSize*array->start);
    addressMap[array] = {addressReg, off};
    return {addressReg, off};
}

PTXregister* KernelGeneratorSimple::loadElement(const cphVBarray* array)
{
    PTXaddress address = calcAddress(array);
    PTXregister* elementReg = registerBank->newRegister(array->type);
    instructionList->add(PTX_LD_GLOBAL, elementReg, address.reg, address.off);
    return elementReg;
}


PTXregister* KernelGeneratorSimple::loadScalar(cphvb_type type,
                                               cphvb_constant value)
{
    PTXkernelParameter* ptxParam = ptxKernel->addParameter(ptxType(type));
    parameters.push_back(KernelParameter(type,value));
    PTXregister* scalarReg = registerBank->newRegister(type);
    instructionList->add(PTX_LD_PARAM, scalarReg, ptxParam);  
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
        dest = registerBank->newRegister(inst->operand[0]->type);
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
        instructionList->add(PTX_ST_GLOBAL, siter->first, 
                             address.reg, address.off);

    }
}

void KernelGeneratorSimple::run(Threads threads,
                                InstructionIterator first,
                                InstructionIterator last)
{
    init(threads);
    for (;first != last; ++first)
    {
        addInstruction(*first);
    }
    storeAll();
    
    KernelShapeSimple* shape = new KernelShapeSimple(threads);
    KernelSimple* kernel = new KernelSimple(ptxKernel,shape);
    kernel->execute(parameters);
    clear();
}
