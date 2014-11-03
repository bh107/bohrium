/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cassert>
#include <stdexcept>
#include "Buffer.hpp"
#include <iostream>

Buffer::Buffer(size_t size_, ResourceManager* resourceManager_) 
    : resourceManager(resourceManager_)
    , device(0)
    , dataType(OCL_UNKNOWN)
    , clBuffer(NULL)
    , writeEvent(resourceManager->completeEvent())
    , size(size_)
{}

Buffer::Buffer(size_t elements, OCLtype dataType_, ResourceManager* resourceManager_) 
    : resourceManager(resourceManager_)
    , device(0)
    , dataType(dataType_)
    , clBuffer(NULL)
    , writeEvent(resourceManager->completeEvent())
    , size(elements * oclSizeOf(dataType))
{}

Buffer::~Buffer()
{
    if (clBuffer != NULL)
    {
        cl::Event::waitForEvents(allEvents());
        delete clBuffer;
    }
}

void Buffer::read(void* hostPtr)
{
    if (clBuffer == NULL)
    {
        clBuffer = resourceManager->createBuffer(size);
    }
    resourceManager->readBuffer(*clBuffer, hostPtr, writeEvent, device);
}

void Buffer::write(void* hostPtr)
{
    if (clBuffer == NULL)
    {
        clBuffer = resourceManager->createBuffer(size);
    }
    writeEvent = resourceManager->enqueueWriteBuffer(*clBuffer, hostPtr, allEvents(), device);
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
    if (clBuffer == NULL)
    {
        clBuffer = resourceManager->createBuffer(size);
    }
    try
    {
        kernel.setArg(argIndex, *clBuffer);
    } catch (cl::Error err)
    {
        std::cerr << "ERROR Setting Buffer kernel arg(" << argIndex << "): " << err.what() << "(" << 
            err.err() << ")" << std::endl;
        throw err;
    }
}

OCLtype Buffer::type() const
{
    return dataType;
}
