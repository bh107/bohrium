/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __BUFFER_HPP
#define __BUFFER_HPP

#include <deque>
#include <CL/cl.hpp>
#include "OCLtype.h"
#include "KernelParameter.hpp"
#include "ResourceManager.hpp"

class Buffer : public KernelParameter
{
private:
    ResourceManager* resourceManager;
    unsigned int device;
    OCLtype dataType;
    cl::Buffer clBuffer;
    cl::Event writeEvent;
    std::deque<cl::Event> readEvents;
    void cleanReadEvents();
protected:
        void printOn(std::ostream& os) const;
public:
    Buffer(size_t size,  ResourceManager* resourceManager);
    Buffer(size_t elements, OCLtype dataType, ResourceManager* resourceManager);
    void read(void* hostPtr);
    void write(void* hostPtr);
    void setWriteEvent(cl::Event);
    cl::Event getWriteEvent();
    void addReadEvent(cl::Event);
    std::deque<cl::Event> getReadEvents();
    std::vector<cl::Event> allEvents();
    void addToKernel(cl::Kernel& kernel, unsigned int argIndex);
    OCLtype type() const;

};


#endif
