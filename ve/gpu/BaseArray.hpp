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

#ifndef __BASEARRAY_HPP
#define __BASEARRAY_HPP

#include "ArrayOperand.hpp"
#include "ResourceManager.hpp"
#include <deque>

#define OCL_BUFFER OCL_TYPES

class BaseArray : public ArrayOperand
{
private:
    ResourceManager* resourceManager;
    OCLtype bufferType;
    //TODO have a map<buffer, device> 
    //and split the array into several buffers
    cl::Buffer buffer;
    unsigned int device;
    bool scalar;
    cl::Event writeEvent;
    std::deque<cl::Event> readEvents;
    void cleanReadEvents();
protected:
public:
    BaseArray(cphvb_array* spec, ResourceManager* resourceManager);
    OCLtype type();
    OCLtype parameterType();
    void sync();
    void setWriteEvent(cl::Event);
    cl::Event getWriteEvent();
    void addReadEvent(cl::Event);
    std::deque<cl::Event> getReadEvents();
    cl::Buffer getBuffer();
    bool isScalar();
    void printKernelParameterType(bool input, std::ostream& source);
    void addToKernel(bool input, cl::Kernel& kernel, unsigned int argIndex) const;
};


#endif
