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
}

template <typename T>
slice<T>& slice<T>::operator[](int rhs)
{
    std::cout << "slice[int] [dim=" << dims << "] " << rhs <<std::endl;
    ranges[dims].begin = rhs;
    ranges[dims].end   = rhs;
    dims++;
    return *this;
}

template <typename T>
slice<T>& slice<T>::operator[](slice_bound rhs)
{
    std::cout << "slice[ALL] [dim=" << dims << "] " << rhs <<std::endl;
    dims++;
    return *this;
}

template <typename T>
slice<T>& slice<T>::operator[](slice_range& rhs)
{
    std::cout << "slice[range] [dim=" << dims << "]" <<std::endl;
    ranges[dims] = rhs;
    dims++;
    return *this;
}

template <typename T>
bh::multi_array<T>& slice<T>::view()
{
    std::cout << " Create the view! " << dims <<std::endl;

    bh_array* array = &storage[op->getKey()];
    array->start        += ranges[1].begin;
    array->stride[1]    += ranges[1].stride;
    array->shape[1]     = array->shape[1] - ranges[1].begin;
    array->shape[1]     = array->shape[1] / ranges[1].stride;

    for(int i=0; i<dims; ++i ) {
        std::cout << "[Dim="<< i << "; " << ranges[i].begin << "," \
                                    << ranges[i].end << "," \
                                    << ranges[i].stride << "]" \
                                    <<std::endl;
    }

    return *op;
}

}

#endif

