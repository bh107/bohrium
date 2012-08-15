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

#include <cassert>
#include <stdexcept>
#include "BaseArray.hpp"

BaseArray::BaseArray(cphvb_array* spec_, ResourceManager* resourceManager) 
    : Buffer(cphvb_nelements(spec_->ndim, spec_->shape), oclType(spec_->type), resourceManager)
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
        if (cphvb_data_malloc(spec) != CPHVB_SUCCESS)
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
    return cphvb_nelements(spec->ndim, spec->shape);
}
