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

#include <queue>
#include <map>
#include <cphvb.h>
#include "KernelGeneratorSimple.hpp"
#include "PTXtype.h"


KernelGeneratorSimple::KernelGeneratorSimple()
{
    registerBank = new RegisterBank();
    offsetMap = createOffsetMap;
}

void KernelGeneratorSimple::init()
{
    threadID = registerBank->newRegister(PTX_UINT32);
    PTXregister* tidx = registerBank->newRegister(PTX_UINT32);
    PTXregister* ntidx = registerBank->newRegister(PTX_UINT32);
    PTXregister* ctaidx = registerBank->newRegister(PTX_UINT32);
    instructionList->add(PTX_MOV, tidx, registerBank->tid_x);
    instructionList->add(PTX_MOV, ntidx, registerBank->ntid_x);
    instructionList->add(PTX_MOV, ctaidx, registerBank->ctaid_x);
    instructionList->add(PTX_MAD, threadID, ntidx, ctaidx, tidx); 
}

void KernelGeneratorSimple::calcOffset(cphVBArray array, PTXregister* reg)
{
    cphvb_intp dim = array->ndim -1;
    cphvb_index dimbound = array->shape[dim];
    PTXregister* tmpReg = registerBank->newRegister(PTX_UINT32);
    instructionList->add(PTX_MOD, tmpReg, threadID, 
                         constantBuffer->newConstant(PTX_UINT32,dimbound));
    instructionList->add(PTX_MUL, offsetReg, tmpReg,  
                  constantBuffer->newConstant(PTX_UINT32,array->stride[dim]));
    for (--dim; dim >= 0; --dim)
    {
        instructionList->add(PTX_DIV, tmpReg, threadID,  
                             constantBuffer->newConstant(PTX_UINT32,dimbound));
        instructionList->add(PTX_MAD, offsetReg, tmpReg,
                    constantBuffer->newConstant(PTX_UINT32,array->stride[dim]),
                             offsetReg);
        dimbound *= array->shape[dim];
    }
}

PTXregister* KernelGeneratorSimple::loadElement(cphVBArray array);
{
    PTXregister* offsetReg = offsetMap->find(array);
    if (offsetReg == NULL)
    {
        offsetReg = registerBank->newRegister(PTX_UINT32);
        calcOffset(array,offsetReg);
        offsetMap->insert(array, offsetReg);
    }
    PTXregister* addressReg;
    int eSize = cphvb_type_size(array->type);
    switch (sizeof(void*))
    {
    case 4:
        addressReg = registerBank->newRegister(PTX_UINT64);
        instructionList->add(PTX_MAD_WIDE, addressReg, offsetReg, 
                   constantBuffer->newConstant(PTX_UINT,eSize),
                   constantBuffer->newConstant(PTX_ADDRESS,array->cudaPtr));
        break;
    case 8:
        addressReg = registerBank->newRegister(PTX_UINT32);
        instructionList->add(PTX_MAD, addressReg, offsetReg, 
                   constantBuffer->newConstant(PTX_UINT,eSize),
                   constantBuffer->newConstant(PTX_ADDRESS,array->cudaPtr));
        break;
    default:
        assert (false);
    }
    PTXregister* elementReg = registerBank->newRegister(array->type);
    instructionList->add(PTX_LOAD, elementReg, addressReg,  
                   constantBuffer->newConstant(PTX_UINT,eSize*array->start));
    return elementReg;
}


PTXregister* KernelGeneratorSimple::loadScalar(cphvb_type type,
                                               cphvb_constant value)
{
    PTXkernelParameter ptxParam = ptxKernel->addParameter(ptxType(type));
    parameters.push_back(KernelParameter(type,value));
    PTXregister* scalarReg = registerBank->newRegister(type);
    instructionList->add(PTX_LD_PARAM, scalarReg, ptxParam,)  
    return scalarReg;
}

void KernelGeneratorSimple::addInstruction(cphVBInstruction* inst)
{
    assert(inst->operand[0] != CPHVB_CONSTANT);
    int nops = cphvb_operands(inst->opcode);
    PTXregister dest;
    PTXregister src[CPHVB_MAX_NO_OPERANDS-1];

    // get register for result
    ElementMap::iterator eiter = elementMap.find(inst->operand[0]);
    if (deiter == elementMap.end())
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
                elementMap[src[i-1]] = operands[i];
            }
            else
            {
                src[i-1] = eiter->second;
            }
        }
        else
        {   //Constant
            src[i-1] = loadScalar(inst->constant[i], 
                                  inst->const_type[i]);
        }
    }
    
    //generate PTX instruction
    generatePTXinstruction(inst->opcode, dest, src);
    
}
