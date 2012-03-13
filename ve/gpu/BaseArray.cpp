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
#include <stdexcept>
#include <cphvb.h>
#include "BaseArray.hpp"

BaseArray::BaseArray(cphvb_array* spec_, ResourceManager* resourceManager_) 
    : ArrayOperand(spec_)
    , resourceManager(resourceManager_)
    , bufferType(oclType(spec_->type))
{
    assert(spec->base == NULL);
    if (!cphvb_scalar(spec))
    {
        scalar = false;
        buffer = resourceManager->createBuffer(size() * oclSizeOf(bufferType));
        device = 0;
        if (spec->data != NULL)
        {
            writeEvent = resourceManager->enqueueWriteBuffer(buffer, spec->data, device);
        } 
        else 
        {
            writeEvent = resourceManager->completeEvent();
        }
    }
    else
    {
        scalar = true;
        writeEvent = resourceManager->completeEvent();
    }
}

void BaseArray::sync()
{
    if (spec->data == NULL)
    {
        if (cphvb_data_malloc(spec) != CPHVB_SUCCESS)
        {
            throw std::runtime_error("Could not allocate memory on host");
        }
    }
    if (!scalar)
        resourceManager->readBuffer(buffer, spec->data, writeEvent, device);
}

OCLtype BaseArray::type()
{
    return bufferType;
}

void BaseArray::setWriteEvent(cl::Event event)
{
    writeEvent = event;
}

cl::Event BaseArray::getWriteEvent()
{
    return writeEvent;
}

void BaseArray::cleanReadEvents()
{
    while (readEvents.front().getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE)
        readEvents.pop_front();
}

void BaseArray::addReadEvent(cl::Event event)
{
    cleanReadEvents();
    readEvents.push_back(event);
}

std::deque<cl::Event> BaseArray::getReadEvents()
{
    cleanReadEvents();
    return readEvents;
}

cl::Buffer BaseArray::getBuffer()
{
    return buffer;
}

bool BaseArray::isScalar()
{
    return scalar;
}

OCLtype BaseArray::parameterType()
{
    if (scalar)
        return bufferType;
    else
        return OCL_BUFFER;
}

void BaseArray::printKernelParameterType(bool input, std::ostream& source)
{
    if (input && scalar)
        source << "const " << oclTypeStr(bufferType);
    else
        source << "__global " << oclTypeStr(bufferType) << "*";
}

void BaseArray::addToKernel(bool input, cl::Kernel& kernel, unsigned int argIndex) const
{
    if (scalar && input)
    {
        assert(spec->data != NULL);
        switch(bufferType)
        {
        case OCL_INT8:
            kernel.setArg(argIndex, *(cl_char*)spec->data);
            break;
        case OCL_INT16:
            kernel.setArg(argIndex, *(cl_short*)spec->data);
            break;
        case OCL_INT32:
            kernel.setArg(argIndex, *(cl_int*)spec->data);
            break;
        case OCL_INT64:
            kernel.setArg(argIndex, *(cl_long*)spec->data);
            break;
        case OCL_UINT8:
            kernel.setArg(argIndex, *(cl_uchar*)spec->data);
            break;
        case OCL_UINT16:
            kernel.setArg(argIndex, *(cl_ushort*)spec->data);
            break;
        case OCL_UINT32:
            kernel.setArg(argIndex, *(cl_uint*)spec->data);
            break;
        case OCL_UINT64:
            kernel.setArg(argIndex, *(cl_ulong*)spec->data);
            break;
        case OCL_FLOAT16:
            kernel.setArg(argIndex, *(cl_half*)spec->data);
            break;
        case OCL_FLOAT32:
            kernel.setArg(argIndex, *(cl_float*)spec->data);
            break;
        case OCL_FLOAT64:
            kernel.setArg(argIndex, *(cl_double*)spec->data);
            break;
        default:
            throw std::runtime_error("Scalar: Unknown type.");
        }    
    } 
    else 
    {
        kernel.setArg(argIndex, buffer);
    }
}
