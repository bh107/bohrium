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
#ifndef __BOHRIUM_BRIDGE_CPP_ITERATOR
#define __BOHRIUM_BRIDGE_CPP_ITERATOR
#include <bh.h>
#include <iterator>

namespace bxx {

template <typename T>
class multi_array_iter : public std::iterator<std::input_iterator_tag, T> {
public:
    // Types
    typedef T  value_type;
    typedef T* pointer;
    typedef T& reference;

    typedef typename multi_array_iter<T>::iterator iterator;

    // Constructors
    multi_array_iter() : data(NULL), offset(NULL) {}

    multi_array_iter(bh_view meta) : view(meta)
    {
        data        = (pointer)(view.base->data);
        offset      = data + view.start;

        last_dim    = view.ndim-1;
        last_e      = bh_nelements(view.ndim, view.shape)-1;
        cur_e       = 0;
        memset(coord, 0, BH_MAXDIM * sizeof(int64_t));
    }

    // Operator overloads
    friend bool operator==(const multi_array_iter& i, const multi_array_iter& j)
    {
        return i.offset == j.offset;
    }

    friend bool operator!=(const multi_array_iter& i, const multi_array_iter& j)
    {
        return i.offset != j.offset;
    }

    multi_array_iter& operator++()      // PREFIX
    {                                   // NOTE: This is extremely inefficient
        cur_e++;
        coord[last_dim]++;

        if (coord[last_dim] >= view.shape[last_dim]) {
            coord[last_dim] = 0;        // Increment coordinates for the remaining dimensions
            for(int64_t j = last_dim-1; j >= 0; --j) {  
                coord[j]++;             // Still within this dimension
                if (coord[j] < view.shape[j]) {      
                    break;
                } else {                // Reached the end of this dimension
                    coord[j] = 0;       // Reset coordinate
                }                       // Loop then continues to increment the next dimension
            }
        }

        if (cur_e > last_e) {
            offset = NULL;
        } else {
            offset = data + view.start;
            for (int64_t j=0; j<=last_dim; ++j) {
                offset += coord[j] * view.stride[j];
            }
        }

        return *this;
    }

    multi_array_iter operator++(int) // postfix
    {
        multi_array_iter result = *this;
        ++(*this);

        return result;
    }

    reference operator*()
    {
        return *offset;
    }

    pointer operator->() {
        return &*offset;
    }

private:
    bh_view view;

    pointer data,
            offset;

    int64_t last_dim,
            last_e;

    int64_t cur_e; 
    int64_t coord[BH_MAXDIM];

};

}
#endif
