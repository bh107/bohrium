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

#include <cassert>
#include <iostream>
#include <sstream>
#include <iomanip> 
#include <stdexcept>
#include "PTXinstruction.hpp"

inline void PTXinstruction::printOpModifierOn(std::ostream& os) const
{
    switch (opcode)
    {
    case PTX_MUL:
    case PTX_MAD:
        if (ptxBaseType(dest->getType()) == PTX_INT || 
            ptxBaseType(dest->getType()) == PTX_UINT)
        {
            
            os << ".lo";
        }
        break;
    case PTX_DIV:
    case PTX_EXP2:
    case PTX_LOG2:
        if (ptxBaseType(dest->getType()) == PTX_FLOAT)
        {
            
            os << ".approx";
        }
        break;
    default:
        break;
    }
}

inline void PTXinstruction::printAritOpOn(std::ostream& os) const
{ 
    switch (opcode)
    {
    case PTX_ADD:
        os << "add";
        break;
    case PTX_SUB:
        os << "sub";
        break;
    case PTX_MUL:
        os << "mul";
        break;
    case PTX_MAD:
        os << "mad";
        break;
    case PTX_MAD_WIDE:
        os << "mad.wide";
        break;
    case PTX_DIV:
        os << "div";
        break;
    case PTX_REM:
        os << "rem";
        break;
    case PTX_EXP2:
        os << "ex2";
        break;
    case PTX_LOG2:
        os << "lg2";
        break;
    case PTX_ABS:
        os << "abs";
        break;
    case PTX_NEG:
        os << "neg";
        break;
    case PTX_SQRT:
        os << "sqrt";
        break;
    case PTX_MOV:
        os << "mov";
        break;
    default:
        assert (false);
     }
    printOpModifierOn(os);
    os << ((opcode==PTX_MAD_WIDE)?ptxWideOpStr(dest->getType()):
        ptxTypeStr(dest->getType())) << " " << *dest;
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        os << ", " << *src[i];
    }
}

inline void PTXinstruction::printRelOpOn(std::ostream& os) const
{ 
    switch (opcode)
    {
    case PTX_SET_EQ:
        os << "set.eq";
        break;
    case PTX_SET_NE:
        os << "set.ne";
        break;
    case PTX_SET_LT:
        os << "set.lt";
        break;
    case PTX_SET_LE:
        os << "set.le";
        break;
    case PTX_SET_GT:
        os << "set.gt";
        break;
    case PTX_SET_GE:
        os << "set.ge";
        break;
    default:
        assert (false);
     }
    os << ptxTypeStr(dest->getType()) << ptxTypeStr(dest->getType()) << 
        " " << *dest;;
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        os << ", " << *src[i];
    }
}

inline void PTXinstruction::printLogicOpOn(std::ostream& os) const
{ 
    PTXregister* srcReg;
    PTXtype type = PTX_INT32;
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        srcReg = dynamic_cast<PTXregister*>(src[i]);
        if (srcReg != NULL)
        {
            type = srcReg->getType();
            break;
        }
    }
    
    switch (opcode)
    {    
    case PTX_SETP_GE:
        os << "setp.ge";
        break;
    default:
        assert (false);
    }
    os <<  ptxTypeStr(type) << " " << *dest << ", " << *src[0] << ", " << 
        *src[1];
}

inline void PTXinstruction::printConvertOpOn(std::ostream& os) const
{
    PTXregister* srcReg = dynamic_cast<PTXregister*>(src[0]);
    assert (srcReg != NULL);
    os << "cvt.rn" << ptxTypeStr(dest->getType()) << "."  << 
        ptxTypeStr(srcReg->getType()) << " " << *dest << ", " << *srcReg;
}

inline void PTXinstruction::printOpOn(std::ostream& os) const 
{
    switch (opcode)
    {
    case PTX_EXIT:
        os << "exit";
        break;
    case PTX_MEMBAR:
        os << "membar.gl";
        break;
    case PTX_BRA:
        os << "bra " << label;
        break;
    case PTX_LD_GLOBAL:
        os << "ld.global" << ptxTypeStr(dest->getType()) << " " << *dest << ", " << 
            "[" << *src[0] << "+" << *src[1] << "]";
        break;
    case PTX_LD_PARAM:
        os << "ld.param" << ptxTypeStr(dest->getType()) << " " << *dest << ", " << 
            "[" << *src[0] << "]";
        break;
    case PTX_ST_GLOBAL:
        os << "st.global" << ptxTypeStr(dest->getType()) <<  " [" << *src[0] << 
            "+" << *src[1] << "], " << *dest; 
        break;
    case PTX_CVT:
        printConvertOpOn(os);
        break;
    case PTX_SETP_GE:
        printLogicOpOn(os);
        break;
    case PTX_SET_EQ:
    case PTX_SET_NE:
    case PTX_SET_LT:
    case PTX_SET_LE:
    case PTX_SET_GT:
    case PTX_SET_GE:
        printRelOpOn(os);
        break;
    default:
        printAritOpOn(os);
        break;
    }
}

inline void PTXinstruction::printOn(std::ostream& os) const 
{
    if (label != NULL && opcode != PTX_BRA)    
    {
        os << label << ":\n";
    }
    if (guard != NULL)
    {
        os << std::setw(6) << (guardMod?"@":"@!") << *guard << " ";
    } 
    else
    {
        os << "          ";
    }
    printOpOn(os);
    os << ";\n";
}

std::ostream& operator<< (std::ostream& os, 
                          PTXinstruction const& ptxInstruction)
{
    ptxInstruction.printOn(os);
    return os;
}

PTXinstruction::PTXinstruction(char* label_,
                               bool guardMod_,
                               PTXregister* guard_,
                               PTXopcode opcode_,
                               PTXregister* dest_,
                               PTXoperand* src_[]) :
    label(label_),
    guardMod(guardMod_),
    guard(guard_),
    opcode(opcode_),
    dest(dest_)
{
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        src[i] = src_[i];
    }
}

PTXinstruction::PTXinstruction(PTXopcode opcode_,
                               PTXregister* dest_,
                               PTXoperand* src_[]) :
    label(NULL),
    guardMod(false),
    guard(NULL),
    opcode(opcode_),
    dest(dest_)
{
    for (int i = 0; i < ptxSrcOperands(opcode); ++i)
    {
        src[i] = src_[i];
    }
}



PTXinstruction::PTXinstruction(PTXopcode opcode_,
                               PTXregister* dest_) :
    label(NULL),
    guardMod(false),
    guard(NULL),
    opcode(opcode_),
    dest(dest_),
    src({NULL,NULL,NULL}) {}

PTXinstruction::PTXinstruction(PTXopcode opcode_,
                               PTXregister* dest_,
                               PTXoperand* src0) :
    label(NULL),
    guardMod(false),
    guard(NULL),
    opcode(opcode_),
    dest(dest_),
    src({src0,NULL,NULL}) {}

PTXinstruction::PTXinstruction(PTXopcode opcode_,
                               PTXregister* dest_,
                               PTXoperand* src0,             
                               PTXoperand* src1) :
    label(NULL),
    guardMod(false),
    guard(NULL),
    opcode(opcode_),
    dest(dest_),
    src({src0,src1,NULL}) {}

PTXinstruction::PTXinstruction(PTXopcode opcode_,
                               PTXregister* dest_,
                               PTXoperand* src0,
                               PTXoperand* src1,             
                               PTXoperand* src2) :
    label(NULL),
    guardMod(false),
    guard(NULL),
    opcode(opcode_),
    dest(dest_),
    src({src0,src1,src2}) {}

PTXinstruction::PTXinstruction(PTXregister* guard_,
                               PTXopcode opcode_,
                               char* label_):
    label(label_),
    guardMod(true),
    guard(guard_),
    opcode(opcode_),
    dest(NULL),
    src({NULL,NULL,NULL}) {}

PTXinstruction::PTXinstruction(PTXregister* guard_,
                               PTXopcode opcode_) :
    label(NULL),
    guardMod(true),
    guard(guard_),
    opcode(opcode_),
    dest(NULL),
    src({NULL,NULL,NULL}) {}

PTXinstruction::PTXinstruction(PTXopcode opcode_) :
    label(NULL),
    guardMod(true),
    guard(NULL),
    opcode(opcode_),
    dest(NULL),
    src({NULL,NULL,NULL}) {}
