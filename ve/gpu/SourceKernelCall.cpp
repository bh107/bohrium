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

#include "SourceKernelCall.hpp"

SourceKernelCall::SourceKernelCall(KernelID id,
                                   std::vector<size_t> shape,
                                   std::string source,
                                   std::vector<KernelParameter*> sizeParameters,
                                   Kernel::Parameters valueParameters)
    : _id(id)
    , _shape(shape)
    , _source(source)
    , _sizeParameters(sizeParameters)
    , _valueParameters(valueParameters) 
{}

KernelID SourceKernelCall::id() const
{
    return _id;
}

size_t SourceKernelCall::functionID() const
{
    return _id.first;
}

size_t SourceKernelCall::literalID() const
{
    return _id.second;
}

std::vector<size_t> SourceKernelCall::shape() const
{
    return _shape;
}

std::string SourceKernelCall::source() const
{
    return _source;
}

Kernel::Parameters SourceKernelCall::valueParameters() const
{
    return _valueParameters;
}
    
Kernel::Parameters SourceKernelCall::allParameters() const
{
    Kernel::Parameters all(_valueParameters);
    for (KernelParameter* kp: _sizeParameters)
        all.push_back(std::make_pair(kp,false));
    return all;
}

void SourceKernelCall::setDiscard(std::set<BaseArray*> discardSet)
{
    _discardSet = discardSet;
}

void SourceKernelCall::addDiscard(BaseArray* array)
{
    _discardSet.insert(array);
}

void SourceKernelCall::deleteBuffers()
{
    for (BaseArray *ba: _discardSet)
    {
        delete ba;
    }
    _discardSet.clear();
}
