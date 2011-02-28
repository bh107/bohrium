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

#ifndef __KERNEL_HPP
#define __KERNEL_HPP

#include <vector>
#include <cuda.h>
#include "KernelParameter.hpp"
#include "KernelShape.hpp"

typedef std::vector<KernelParameter> ParameterList;
typedef std::vector<PTXtype> Signature;

class Kernel
{
private:
    void setBlockShape(int x, int y, int z);
protected:
    CUmodule module;
    CUfunction entry;
    Signature signature;
    Kernel(CUmodule module,
           CUfunction entry,
           Signature signature);
    void setParameters(ParameterList parameters);
    void launchGrid(KernelShape shape);
public:
    virtual void execute(ParameterList parameters) = 0;
};

#endif
