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
#ifndef __BOHRIUM_BRIDGE_CPP_ITERATOR
#define __BOHRIUM_BRIDGE_CPP_ITERATOR
#include "bh.h"
#include <iterator>

namespace bh {

template <typename T>
class Vector_iter : public std::iterator<std::input_iterator_tag, T> {
public:
    // Types
    typedef T  value_type;
    typedef T* pointer;
    typedef T& reference;

    typedef typename Vector_iter<T>::iterator iterator;

    // Constructors
    Vector_iter() : data(NULL) {}

    Vector_iter(bh_array x) : operand(x) {

        data        = (pointer)bh_base_array( &operand )->data;
        last_dim    = operand.ndim-1;
        last_e      = bh_nelements(operand.ndim, operand.shape )-1;
        cur_e       = 0;
        offset      = operand.start;

        memset(coord, 0, BH_MAXDIM * sizeof(bh_index));

    }

    // Operator overloads
    friend bool operator==(const Vector_iter& i, const Vector_iter& j)
    {
        return i->data == j->data;
    }

    friend bool operator!=(const Vector_iter& i, const Vector_iter& j)
    {
        return i.data != j.data;
    }

    Vector_iter& operator++()   // prefix
    {
        data++;
        cur_e++;
        if (cur_e == last_e) {
            data = NULL;
        }
        return *this;
    }

    Vector_iter operator++(int) // postfix
    {
        data++;
        cur_e++;
        if (cur_e == last_e) {
            data = NULL;
        }
        return *this;
    }

    reference operator*()
    {
        return *data;
    }

    pointer operator->() {
        return &*data;
    }

private:

    pointer data;
    bh_array operand;

    bh_index    offset,
                last_dim,
                last_e,
                k;

    bh_index cur_e; 
    bh_index coord[BH_MAXDIM];

};

}
#endif
