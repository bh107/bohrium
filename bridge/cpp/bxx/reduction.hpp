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

namespace bxx {

template <typename T_out, typename T_in>
inline
multi_array<T_out>& create_reduce_result(multi_array<T_in>& lhs, int64_t rhs)
{
    // Construct result array
    multi_array<T_out>* result = &Runtime::instance().create<T_out>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {               // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    return *result;
}

//
// Partial reduction
template <typename T>
multi_array<T>& reduce_add(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_add_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_mul(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_multiply_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_min(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_minimum_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_max(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_maximum_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_and(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_logical_and_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_or(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_logical_or_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_xor(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_logical_xor_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_bw_and(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_bitwise_and_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_bw_or(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_bitwise_or_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& reduce_bw_xor(multi_array<T>& op, int64_t axis)
{
    multi_array<T>* result = &create_reduce_result<T,T>(op, axis);
    bh_bitwise_xor_reduce(*result, op, axis);
    result->setTemp(true);
    return *result;
}


//
// Complete reduction
template <typename T>
multi_array<T>& sum(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* in = &op;
    multi_array<T>* result = &create_reduce_result<T,T>(op, 0);
    bh_add_reduce(*result, *in, (int64_t)0);
    
    for(size_t i=1; i<dims; i++) {
        in = result;
        in->setTemp(true);
        result = &create_reduce_result<T,T>(*in, 0);
        bh_add_reduce(*result, *in, (int64_t)0);
    }
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& product(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* in = &op;
    multi_array<T>* result = &create_reduce_result<T,T>(op, 0);
    bh_add_reduce(*result, *in, (int64_t)0);
    
    for(size_t i=1; i<dims; i++) {
        in = result;
        in->setTemp(true);
        result = &create_reduce_result<T,T>(*in, 0);
        bh_multiply_reduce(*result, *in, (int64_t)0);
    }
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& min(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* in = &op;
    multi_array<T>* result = &create_reduce_result<T,T>(op, 0);
    bh_add_reduce(*result, *in, (int64_t)0);
    
    for(size_t i=1; i<dims; i++) {
        in = result;
        in->setTemp(true);
        result = &create_reduce_result<T,T>(*in, 0);
        bh_minimum_reduce(*result, *in, (int64_t)0);
    }
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& max(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* in = &op;
    multi_array<T>* result = &create_reduce_result<T,T>(op, 0);
    bh_add_reduce(*result, *in, (int64_t)0);
    
    for(size_t i=1; i<dims; i++) {
        in = result;
        in->setTemp(true);
        result = &create_reduce_result<T,T>(*in, 0);
        bh_maximum_reduce(*result, *in, (int64_t)0);
    }
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& any(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* in = &op;
    multi_array<T>* result = &create_reduce_result<T,T>(op, 0);
    bh_add_reduce(*result, *in, (int64_t)0);
    
    for(size_t i=1; i<dims; i++) {
        in = result;
        in->setTemp(true);
        result = &create_reduce_result<T,T>(*in, 0);
        bh_logical_or_reduce(*result, *in, (int64_t)0);
    }
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& all(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* in = &op;
    multi_array<T>* result = &create_reduce_result<T,T>(op, 0);
    bh_add_reduce(*result, *in, (int64_t)0);
    
    for(size_t i=1; i<dims; i++) {
        in = result;
        in->setTemp(true);
        result = &create_reduce_result<T,T>(*in, 0);
        bh_logical_and_reduce(*result, *in, (int64_t)0);
    }
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<size_t>& count(multi_array<T>& op)
{
    return sum(op.template as<size_t>());
}

}
#endif

