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

#ifndef __BUFFER_HPP
#define __BUFFER_HPP

#include <deque>
#include <CL/cl.hpp>
#include "ResourceManager.hpp"

class Buffer
{
private:
    ResourceManager* resourceManager;
    unsigned int device;
    cl::Buffer buffer;
    cl::Event writeEvent;
    std::deque<cl::Event> readEvents;
    void cleanReadEvents();
public:
    Buffer(size_t size, ResourceManager* resourceManager);
    void read(void* hostPtr);
    void write(void* hostPtr);
    void setWriteEvent(cl::Event);
    cl::Event getWriteEvent();
    void addReadEvent(cl::Event);
    std::deque<cl::Event> getReadEvents();
    std::vector<cl::Event> allEvents();
    cl::Buffer clBuffer();
};


#endif
