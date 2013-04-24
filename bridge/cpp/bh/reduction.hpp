/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

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
#ifndef __BOHRIUM_BRIDGE_CPP_REDUCTION
#define __BOHRIUM_BRIDGE_CPP_REDUCTION

namespace bh {

bh_opcode reducible_to_opcode(reducible opcode)
{
    switch(opcode) {
        case ADD:
            return BH_ADD_REDUCE;
            break;
        case MULTIPLY:
            return BH_MULTIPLY_REDUCE;
            break;
        case MIN:
            return BH_MINIMUM_REDUCE;
            break;
        case MAX:
            return BH_MAXIMUM_REDUCE;
            break;
        case LOGICAL_AND:
            return BH_LOGICAL_AND_REDUCE;
            break;
        case LOGICAL_OR:
            return BH_LOGICAL_OR_REDUCE;
            break;
        case BITWISE_AND:
            return BH_BITWISE_AND_REDUCE;
            break;
        case BITWISE_OR:
            return BH_BITWISE_OR_REDUCE;
            break;
        default:
            throw std::runtime_error("Error: Unsupported opcode for reduction.\n");
    }
}

template <typename T>
multi_array<T>& multi_array<T>::reduce(reducible opcode, unsigned int axis)
{
    multi_array<T>* result = new multi_array<T>();
    result->setTemp(true);

    storage[result->getKey()].base  = NULL;
    storage[result->getKey()].data  = NULL;
    storage[result->getKey()].start = 0;

    if (storage[key].ndim == 1) {               // Reduce to "scalar", 1d with one element
        storage[result->getKey()].ndim      = 1;
        storage[result->getKey()].shape[0]  = 1;
        storage[result->getKey()].stride[0] = storage[key].stride[0];
    } else {                                    // Reduce dimensionality by one
        storage[result->getKey()].ndim  = storage[key].ndim -1;
        for(bh_index i=0; i< storage[result->getKey()].ndim; i++) {
            storage[result->getKey()].shape[i] = storage[key].shape[i+1];
        }
        for(bh_index i=0; i< storage[result->getKey()].ndim; i++) {
            storage[result->getKey()].stride[i] = storage[key].stride[i+1];
        }
    }

    Runtime::instance()->enqueue(reducible_to_opcode(opcode), *result, *this, axis);

    return *result;
}

template <typename T>
multi_array<T>& multi_array<T>::sum()
{
    multi_array<T>* result = new multi_array<T>();
    result->setTemp(true);

    return *result;
}

template <typename T>
multi_array<T>& multi_array<T>::product()
{
    multi_array<T>* result = new multi_array<T>();
    result->setTemp(true);

    return *result;
}

template <typename T>
multi_array<T>& multi_array<T>::min()
{
    multi_array<T>* result = new multi_array<T>();
    result->setTemp(true);

    return *result;
}

template <typename T>
multi_array<T>& multi_array<T>::max()
{
    multi_array<T>* result = new multi_array<T>();
    result->setTemp(true);

    return *result;
}

template <typename T>
multi_array<bool>& multi_array<T>::any()
{
    multi_array<T>* result = new multi_array<T>();
    result->setTemp(true);

    return *result;
}

template <typename T>
multi_array<bool>& multi_array<T>::all()
{
    multi_array<T>* result = new multi_array<T>();
    result->setTemp(true);

    return *result;
}

template <typename T>
multi_array<size_t>& multi_array<T>::count()
{
    multi_array<T>* result = new multi_array<T>();
    result->setTemp(true);

    return *result;
}

}
#endif

