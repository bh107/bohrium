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

#include <cassert>
#include <stdexcept>
#include "Buffer.hpp"

Buffer::Buffer(size_t size, ResourceManager* resourceManager_) 
    : resourceManager(resourceManager_)
    , device(0)
    , dataType(OCL_UNKNOWN)
    , clBuffer(resourceManager->createBuffer(size))
    , writeEvent(resourceManager->completeEvent())
{}

Buffer::Buffer(size_t elements, OCLtype dataType_, ResourceManager* resourceManager_) 
    : resourceManager(resourceManager_)
    , device(0)
    , dataType(dataType_)
    , clBuffer(resourceManager->createBuffer(elements * oclSizeOf(dataType)))
    , writeEvent(resourceManager->completeEvent())
{}

void Buffer::read(void* hostPtr)
{
    resourceManager->readBuffer(clBuffer, hostPtr, writeEvent, device);
}

void Buffer::write(void* hostPtr)
{
    writeEvent = resourceManager->enqueueWriteBuffer(clBuffer, hostPtr, allEvents(), device);
}

void Buffer::setWriteEvent(cl::Event event)
{
    writeEvent = event;
}

cl::Event Buffer::getWriteEvent()
{
    return writeEvent;
}

void Buffer::cleanReadEvents()
{
    while (!readEvents.empty() && readEvents.front().getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE)
        readEvents.pop_front();
}

void Buffer::addReadEvent(cl::Event event)
{
    cleanReadEvents();
    readEvents.push_back(event);
}

std::deque<cl::Event> Buffer::getReadEvents()
{
    cleanReadEvents();
    return readEvents;
}

std::vector<cl::Event> Buffer::allEvents()
{
    cleanReadEvents();
    std::vector<cl::Event> res(readEvents.begin(), readEvents.end());
    res.push_back(writeEvent);
    return res;
}

void Buffer::printOn(std::ostream& os) const
{
    os << "__global " << oclTypeStr(dataType) << "*";
}

void Buffer::addToKernel(cl::Kernel& kernel, unsigned int argIndex)
{
    try
    {
        kernel.setArg(argIndex, clBuffer);
    } catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        throw err;
    }
}

OCLtype Buffer::type() const
{
    return dataType;
}
