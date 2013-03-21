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
multi_array<T>& multi_array<T>::operator=(slice<T>& rhs) {
    std::cout << " Calling stuff! " << std::endl;
    multi_array<T>* vv = &rhs.view();
    bh_pprint_array(&storage[vv->getKey()]);
    storage[getKey()] = storage[vv->getKey()];
    return *this;
}

template <typename T>
bh::multi_array<T>& slice<T>::view()
{
    std::cout << " Create the view! " << dims <<std::endl;
    multi_array<T>* alias = &Runtime::instance()->view(*op);

    bh_array* rhs = &storage[op->getKey()];
    bh_array* lhs = &storage[alias->getKey()];

    lhs->start  = rhs->start;
    lhs->ndim   = rhs->ndim;
    for(int i=0; i<dims; ++i ) {
        std::cout << "[Dim="<< i << "; " << ranges[i].begin << "," \
                                    << ranges[i].end << "," \
                                    << ranges[i].stride << "]" \
                                    << std::endl;

        lhs->start        += rhs->stride[i] * ranges[i].begin;
        lhs->stride[i]    = rhs->stride[i] * ranges[i].stride;
        
        lhs->shape[i] = ranges[i].begin;
        if (ranges[i].end < 0) {
            lhs->shape[i] = rhs->shape[i] + ranges[i].end;
        } else {
            lhs->shape[i] = ranges[i].end;
        }
    }
    std::cout << ">>>>>" << std::endl;
    bh_pprint_array(lhs);
    std::cout << "<<<<<" << std::endl;
    return *alias;
}

}

#endif

