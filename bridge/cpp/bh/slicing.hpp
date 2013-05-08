/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
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
#ifndef __BOHRIUM_BRIDGE_CPP_SLICING
#define __BOHRIUM_BRIDGE_CPP_SLICING

namespace bh {

slice_range::slice_range() : begin(0), end(-1), stride(1) {}
slice_range::slice_range(int begin, int end, unsigned int stride) : begin(begin), end(end), stride(stride) {}

slice_range& _(int begin, int end, unsigned int stride)
{
    return *(new slice_range(begin, end, stride));
}


template <typename T>
slice<T>::slice(multi_array<T>& op) : op(&op), dims(0)
{
    for(int i=0; i<BH_MAXDIM; i++) {
        ranges[i] = slice_range();
    }
}

template <typename T>
slice<T>& slice<T>::operator[](int rhs)
{
    ranges[dims].begin = rhs;
    ranges[dims].end   = rhs;
    dims++;
    return *this;
}

template <typename T>
slice<T>& slice<T>::operator[](slice_range& rhs)
{
    ranges[dims] = rhs;
    dims++;
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator=(slice<T>& rhs) {
    multi_array<T>* vv = &rhs.view();
    storage[getKey()] = storage[vv->getKey()];
    return *this;
}

template <typename T>
multi_array<T>& slice<T>::operator=(T rhs) {
    multi_array<T>* vv = &this->view();
    *vv = rhs;
    return *vv;
}

template <typename T>
bh::multi_array<T>& slice<T>::view()
{
    multi_array<T>* alias = &Runtime::instance()->view(*op);

    bh_array* rhs = &storage[op->getKey()];     // The operand getting sliced
    bh_array* lhs = &storage[alias->getKey()];  // The view as a result of slicing

    lhs->ndim   = rhs->ndim;                    // Rank is maintained
    lhs->start  = rhs->start;                   // Start is initialy the same
    int b, e;

    for(int i=rhs->ndim-1; i >= 0; --i ) {
                                                // Compute the "[beginning, end[" indexes
        b = ranges[i].begin < 0 ? rhs->shape[i] + ranges[i].begin : ranges[i].begin;
        e = ranges[i].end   < 0 ? rhs->shape[i] + ranges[i].end   : ranges[i].end;

        if (b<=e) {                             // Ensure that the range is valid
            lhs->start      += b * rhs->stride[i];
            lhs->shape[i]   = 1 + (((e-b) - 1) / ranges[i].stride); // ceil
            lhs->stride[i]  = ranges[i].stride * rhs->stride[i];
        } else {
            throw std::runtime_error("Invalid range.");
        }
    }

    return *alias;
}

}

#endif

