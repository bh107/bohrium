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

#include <iostream>
#include <iomanip>
#include "PTXkernel.hpp"


PTXkernel::PTXkernel(PTXversion version_,
                     CUDAtarget target_,
                     PTXregisterBank* registerBank_,
                     PTXinstructionList* instructionList_) :
    version(version_),
    target(target_),
    parameterList(new PTXparameterList(128)),
    parameterCount(0),
    registerBank(registerBank_),
    instructionList(instructionList_) {}

void PTXkernel::clear()
{
    parameterCount = 0;
    parameterList->clear();
    //registerBank and instructioList is cleared form KernelGenerator
}

PTXkernelParameter* PTXkernel::addParameter(PTXtype type)
{
    return parameterList->next(type, parameterCount++);
}

inline void PTXkernel::printOn(std::ostream& os) const
{
    os << ".version " << ptxVersionStr(version) << "\n";
    os << ".target "<< cudaTargetStr(target) << "\n";
    os << ".entry " << std::setw(10) << name << " (";
    PTXparameterList::iterator piter = parameterList->begin();
    if (piter != parameterList->end())
    {
        os <<= *piter;
        for (++piter; piter != parameterList->end(); ++piter)
        {
            os << ",\n" << std::setw(21) << " ";
            os <<= *piter;
        }
    }
    os << ")\n{\n";
    os <<= *registerBank;
    PTXinstructionList::iterator iter = instructionList->begin();
    for(;iter != instructionList->end(); ++iter)
    {
        os << *iter;
    }
    os << "}\n";
}

std::ostream& operator<< (std::ostream& os, 
                          PTXkernel const& ptxKernel)
{
    ptxKernel.printOn(os);
    return os;
}

Signature PTXkernel::getSignature()
{
    Signature sig;
    PTXparameterList::iterator iter = parameterList->begin();
    for (; iter != parameterList->end(); ++iter)
    {
        sig.push_back(iter->getType());
    }
    return sig;
}

