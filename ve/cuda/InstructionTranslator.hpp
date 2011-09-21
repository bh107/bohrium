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

#ifndef __INSTRUCTIONTRANSLATORSIMPLE_HPP
#define __INSTRUCTIONTRANSLATORSIMPLE_HPP

#include "InstructionTranslator.hpp"
#include "PTXinstruction.hpp"
#include "PTXregisterBank.hpp"

class InstructionTranslator
{
private:
    PTXinstructionList* instructionList;
    PTXregisterBank* registerBank;
    PTXregister* convert(PTXtype resType, 
                         PTXregister* src);
public:
    InstructionTranslator(PTXinstructionList* instuctionList,
                          PTXregisterBank* registerBank);
    void translate(cphvb_opcode vbOpcode, 
                   PTXregister* dest, 
                   PTXregister* src[]);
};

#endif
