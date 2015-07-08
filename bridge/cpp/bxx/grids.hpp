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
#ifndef __BOHRIUM_BRIDGE_CPP_GRIDS
#define __BOHRIUM_BRIDGE_CPP_GRIDS

namespace bxx {

template <typename T>
inline
multi_array<T>& gridify(multi_array<T>& x, uint64_t axis)
{
    if (x.getRank() != 1) {
        throw std::runtime_error("Input-array must be 1D.\n");
    }
    if (!((axis == 1) || (axis == 0))) {
        throw std::runtime_error("Axis must be 0 or 1.\n");
    }

    multi_array<T> b;                   // Create broadcast view
    b = x;
    b.meta.ndim = 2;
    b.meta.shape[1] = b.meta.shape[0];
    b.meta.stride[0] = 0 == axis;
    b.meta.stride[1] = 0 != axis;

    return copy(b);                     // Construct a copy
}

}
#endif

