/*
 * Copyright 2012 Troels Blum <troels@blum.dk>
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

#ifndef __BASEARRAY_HPP
#define __BASEARRAY_HPP

#include "Buffer.hpp"
#include "KernelParameter.hpp"

class BaseArray : public KernelParameter
{
private:
    cphvb_array* spec;
    OCLtype bufferType;
    Buffer buffer;
protected:
    void printOn(std::ostream& os) const;
public:
    BaseArray(cphvb_array* spec, ResourceManager* resourceManager);
    OCLtype type() const;
    void sync();
    void addToKernel(cl::Kernel& kernel, unsigned int argIndex) const;
};


#endif
