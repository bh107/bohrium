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

#ifndef __PTXKERNEL_HPP
#define __PTXKERNEL_HPP

#include <vector>
#include "PTXkernelParameter.hpp"
#include "PTXregisterBank.hpp"
#include "PTXkernelBody.hpp"

typedef std::vector<PTXkernelParameter> PTXparameterList;

enum PTXversion 
{
    ISA_14,
    ISA_22
};

enum CUDAtarget
{
    SM_10,
    SM_11,
    SM_12,
    SM_13,
    SM_20,
};

class PTXkernel
{
private:
    PTXversion version;
    CUDAtarget target;
    char name[128];
    PTXparameterList parameterList;
    PTXregisterBank* registerBank;
    PTXkernelBody* kernelBody;
public:
    PTXkernel(PTXversion version,
                    CUDAtarget target,
                    PTXregisterBank* registerBank,
                    PTXkernelBody* kernelBody);
    int snprint(char* buf, int size);
};

#endif
