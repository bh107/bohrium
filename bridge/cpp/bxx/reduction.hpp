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

template <typename T>
multi_array<T>& sum(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* result = &bh_add_reduce(op, (int64_t)0);
    for(size_t i=1; i<dims; i++) {
        result = &bh_add_reduce(*result, (int64_t)0);
    }

    return *result;
}

template <typename T>
multi_array<T>& product(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* result = &bh_multiply_reduce(op, (int64_t)0);
    for(size_t i=1; i<dims; i++) {
        result = &bh_multiply_reduce(*result, (int64_t)0);
    }

    return *result;
}

template <typename T>
multi_array<T>& min(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* result = &bh_minimum_reduce(op, (int64_t)0);
    for(size_t i=1; i<dims; i++) {
        result = &bh_minimum_reduce(*result, (int64_t)0);
    }

    return *result;
}

template <typename T>
multi_array<T>& max(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* result = &bh_maximum_reduce(op, (int64_t)0);
    for(size_t i=1; i<dims; i++) {
        result = &bh_maximum_reduce(*result, (int64_t)0);
    }

    return *result;
}

template <typename T>
multi_array<bool>& any(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* result = &bh_logical_or_reduce(op, (int64_t)0);
    for(size_t i=1; i<dims; i++) {
        result = &bh_logical_or_reduce(*result, (int64_t)0);
    }

    return *result;
}

template <typename T>
multi_array<bool>& all(multi_array<T>& op)
{
    size_t dims = op.meta.ndim;

    multi_array<T>* result = &bh_logical_and_reduce(op, (int64_t)0);
    for(size_t i=1; i<dims; i++) {
        result = &bh_logical_and_reduce(*result, (int64_t)0);
    }
    return *result;
}

template <typename T>
multi_array<size_t>& count(multi_array<T>& op)
{
    return sum(op.template as<size_t>());
}

}
#endif

