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

#ifndef __PTXKERNEL_HPP
#define __PTXKERNEL_HPP

#include <vector>
#include <StaticContainer.hpp>
#include "PTXversion.h"
#include "PTXkernelParameter.hpp"
#include "PTXregisterBank.hpp"
#include "PTXinstruction.hpp"

typedef std::vector<PTXtype> Signature;
typedef StaticContainer<PTXkernelParameter> PTXparameterList;

class PTXkernel
{
private:
    PTXversion version;
    CUDAtarget target;
    PTXparameterList* parameterList;
    int parameterCount;
    PTXregisterBank* registerBank;
    PTXinstructionList* instructionList;
protected:
    void printOn(std::ostream& os) const;
public:
    char name[128];
    PTXkernel(PTXversion version,
              CUDAtarget target,
              PTXregisterBank* registerBank,
              PTXinstructionList* instructionList);
    void clear();
    PTXkernelParameter* addParameter(PTXtype type);
    Signature getSignature();
    friend std::ostream& operator<< (std::ostream& os, 
                                     PTXkernel const& ptxKernel);
};

#endif
