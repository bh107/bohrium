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

#ifndef __RANDOMNUMBERGENERATORHYBRIDTAUS_HPP
#define __RANDOMNUMBERGENERATORHYBRIDTAUS_HPP

#include <cuda.h>
#include "KernelPredefined.hpp"
#include "RandomNumberGenerator.hpp"


#define HT_TPB (32)
#define HT_BPG (192)

//typedef unsigned int uint;

class RandomNumberGeneratorHybridTaus : public RandomNumberGenerator
{
private:
    KernelPredefined* htrand_float32;
    KernelPredefined* htrand_uint32;
    KernelPredefined* htrand_int32;
    KernelPredefined* htrand_float32_step;
    KernelPredefined* htrand_uint32_step;
    KernelPredefined* htrand_int32_step;
    KernelShape* shape;
    uint state[HT_TPB*HT_BPG][4];
    CUdeviceptr cudaState;
    void initState();
public:
    RandomNumberGeneratorHybridTaus();
    ~RandomNumberGeneratorHybridTaus();
    void fill(cphVBarray* array);
};

#endif
