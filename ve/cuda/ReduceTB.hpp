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

#ifndef __REDUCETB_HPP
#define __REDUCETB_HPP

#include <cuda.h>
#include "cphVBarray.hpp"
#include "KernelPredefined.hpp"

#define R_TPB (128)
#define R_BPG (192)

//typedef unsigned int uint;

class ReduceTB
{
private:
    KernelPredefined* add_reduce_float_1d;
    KernelShape* shape;
    float hostRes[R_TPB*R_BPG];
    CUdeviceptr cudaRes;
    void initState();
public:
    ReduceTB();
    ~ReduceTB();
    void reduce(cphVBarray* resArray, cphVBarray* array);
};

#endif
