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

#ifndef __SOURCEKERNELCALL
#define __SOURCEKERNELCALL

#include <utility>
#include <string>
#include <vector>
#include <set>
#include "Kernel.hpp"

typedef std::pair<size_t,size_t> KernelID;

class SourceKernelCall
{
private:
    KernelID _id;
    std::vector<size_t> _shape;
    std::string _source;
    std::vector<KernelParameter*> _sizeParameters;
    Kernel::Parameters _valueParameters;
    std::set<BaseArray*> _discardSet;
public:
    SourceKernelCall(KernelID id,
                     std::vector<size_t> shape,
                     std::string source,
                     std::vector<KernelParameter*> sizeParameters,
                     Kernel::Parameters valueParameters);
    KernelID id() const;
    size_t functionID() const;
    size_t literalID() const;
    std::vector<size_t> shape() const;
    std::string source() const;
    Kernel::Parameters valueParameters() const;    
    Kernel::Parameters allParameters() const;
    void setDiscard(std::set<BaseArray*> discardSet);
    void addDiscard(BaseArray* array);
    void deleteBuffers();
};

#endif
