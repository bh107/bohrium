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

#include <stdexcept>
#include "InstructionTranslator.hpp"

InstructionTranslator::InstructionTranslator(
    PTXinstructionList* instructionList_,
    PTXregisterBank* registerBank_) :
    instructionList(instructionList_),
    registerBank(registerBank_) {}
    

PTXregister* InstructionTranslator::convert(PTXtype resType, 
                                            PTXregister* src)
{
    throw std::runtime_error("Can not convert types.");
    return src;
}

void InstructionTranslator::translate(cphvb_opcode vbOpcode, 
                                      PTXregister* dest, 
                                      PTXregister* src[])
{
    PTXtype resType = dest->type;
    PTXoperand* mySrc[CPHVB_MAX_NO_OPERANDS-1];
    for (int i = 0; i < cphvb_operands(vbOpcode) -1; ++i)
    {
        if (src[i]->type != resType)
        {
            mySrc[i] = convert(resType, src[i]);
        }
        else
        {
            mySrc[i] = src[i];
        }
    }
    PTXopcode ptxOpcode = ptxOpcodeMap(vbOpcode);
    if (ptxOpcode < 0)
    {
        throw std::runtime_error("Operation not supported.");
    }
    instructionList->add(ptxOpcode, dest, mySrc);
}
