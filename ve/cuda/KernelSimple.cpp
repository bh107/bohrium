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

#include "Kernel.hpp"
#include "KernelSimple.hpp"

KernelSimple::KernelSimple(CUmodule module_, 
                           CUfunction entry_, 
                           Signature signature_,
                           KernelShape shape_) :
    Kernel(module_, entry_, signature_),
    shape(shape_) {}

void KernelSimple::execute(ParameterList parameters)
{
    setParameters(parameters);
    setBlockShape(shape.threadsPerBlockX, 
                  shape.threadsPerBlockY, shape.threadsPerBlockY);
    launchGrid(shape.blocksPerGridX, shape.blocksPerGridY);
}
