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
multi_array<T>& reduce(multi_array<T>& op, reducible opcode, size_t axis)
{
    multi_array<T>* result = &Runtime::instance().temp<T>();

    bh_array* res_a = &Runtime::instance().storage[result->getKey()];
    bh_array* op_a  = &Runtime::instance().storage[op.getKey()];

    res_a->base  = NULL;
    res_a->data  = NULL;
    res_a->start = 0;

    if (op_a->ndim == 1) {                  // Pseudo-scalar; one element
        res_a->ndim      = 1;
        res_a->shape[0]  = 1;
        res_a->stride[0] = op_a->stride[0];
    } else {                                // Remove axis
        res_a->ndim  = op_a->ndim -1;
        int64_t stride = 1; 
        for(int64_t i=op_a->ndim-1, j=res_a->ndim-1; i>=0; --i) {
            if (i!=(int64_t)axis) {
                res_a->shape[j]  = op_a->shape[i];
                res_a->stride[j] = stride;
                stride *= res_a->shape[j];
                --j;
            }
        }
    }

    Runtime::instance().enqueue(reducible_to_opcode(opcode), *result, op, (bh_int64)axis);

    return *result;
}

template <typename T>
multi_array<T>& sum(multi_array<T>& op)
{
    size_t dims = Runtime::instance().storage[op.getKey()].ndim;

    multi_array<T>* result = &reduce(op, ADD, 0);
    for(size_t i=1; i<dims; i++) {
        result = &reduce(*result, ADD, 0);
    }
    return *result;
}

template <typename T>
multi_array<T>& product(multi_array<T>& op)
{
    size_t dims = Runtime::instance().storage[op.getKey()].ndim;

    multi_array<T>* result = &reduce(op, MULTIPLY, 0);
    for(size_t i=1; i<dims; i++) {
        result = &reduce(*result, MULTIPLY, 0);
    }

    return *result;
}

template <typename T>
multi_array<T>& min(multi_array<T>& op)
{
    size_t dims = Runtime::instance().storage[op.getKey()].ndim;

    multi_array<T>* result = &reduce(op, MIN, 0);
    for(size_t i=1; i<dims; i++) {
        result = &reduce(*result, MIN, 0);
    }
    return *result;
}

template <typename T>
multi_array<T>& max(multi_array<T>& op)
{
    size_t dims = Runtime::instance().storage[op.getKey()].ndim;

    multi_array<T>* result = &reduce(op, MAX, 0);
    for(size_t i=1; i<dims; i++) {
        result = &reduce(*result, MAX, 0);
    }
    return *result;
}

template <typename T>
multi_array<bool>& any(multi_array<T>& op)
{
    size_t dims = Runtime::instance().storage[op.getKey()].ndim;

    multi_array<T>* result = &reduce(op, LOGICAL_OR, 0);
    for(size_t i=1; i<dims; i++) {
        result = &reduce(*result, LOGICAL_OR, 0);
    }
    return *result;
}

template <typename T>
multi_array<bool>& all(multi_array<T>& op)
{
    size_t dims = Runtime::instance().storage[op.getKey()].ndim;

    multi_array<T>* result = &reduce(op, LOGICAL_AND, 0);
    for(size_t i=1; i<dims; i++) {
        result = &reduce(*result, LOGICAL_AND, 0);
    }
    return *result;
}

template <typename T>
multi_array<size_t>& count(multi_array<T>& op)
{
    multi_array<T>* result = &Runtime::instance().temp<T>();
    result->setTemp(true);

    result = &sum(op.template as<size_t>());

    return *result;
}

}
#endif

