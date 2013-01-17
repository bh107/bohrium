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
#include "BaseArray.hpp"

BaseArray::BaseArray(bh_array* spec_, ResourceManager* resourceManager) 
    : Buffer(bh_nelements(spec_->ndim, spec_->shape), oclType(spec_->type), resourceManager)
    , spec(spec_)
{
    assert(spec->base == NULL);
    if (spec->data != NULL)
    {
        write(spec->data);
    } 
}

void BaseArray::sync()
{
    if (spec->data == NULL)
    {
        if (bh_data_malloc(spec) != CPHVB_SUCCESS)
        {
            throw std::runtime_error("Could not allocate memory on host");
        }
    }
    read(spec->data);
}

void BaseArray::update()
{
    assert(spec->data != NULL);
    write(spec->data);
}

size_t BaseArray::size()
{
    return bh_nelements(spec->ndim, spec->shape);
}
