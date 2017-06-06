/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

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

#ifndef __BHXX_MULTI_ARRAY_H
#define __BHXX_MULTI_ARRAY_H

#include "util.hpp"
#include <bh_component.hpp>
#include <bxx/traits.hpp>
#include <ostream>
#include <vector>

namespace bhxx {

// The base underlying (multiple) arrays
template <typename T>
class BhBase {
  public:
    // NB: `base` must survive until the next flush and not when the `BhBase` object is
    // freed
    bh_base*  base;
    typedef T scalar_type;
    BhBase(size_t nelem) : base(new bh_base()) {
        base->data  = nullptr;
        base->nelem = nelem;
        bxx::assign_array_type<T>(base);
        //        std::cout << "Create base " << this << std::endl;
    }

    ~BhBase();
};

template <typename T>
class BhArray {
  public:
    // The data type of each array element
    typedef T scalar_type;
    // The array offset (from the start of the base in number of elements)
    size_t offset;
    // The array shape (size of each dimension in number of elements)
    Shape shape;
    // The array stride (the absolute stride of each dimension in number of elements)
    Stride stride;
    // Pointer to the base of this array
    std::shared_ptr<BhBase<T>> base;

    // Create a new view that points to a new base
    BhArray(const Shape& shape, const Stride& stride, const size_t offset = 0)
          : offset(offset),
            shape(shape),
            stride(stride),
            base(new BhBase<T>(shape.prod())) {}

    // Create a new view that points to a new base (contiguous stride, row-major)
    BhArray(const Shape& shape, const size_t offset = 0)
          : offset(offset),
            shape(shape),
            stride(contiguous_stride(shape)),
            base(new BhBase<T>(shape.prod())) {}

    // Create a view that points to the given base
    BhArray(const std::shared_ptr<BhBase<T>>& base, const Shape& shape,
            const Stride& stride, const size_t offset = 0)
          : offset(offset), shape(shape), stride(stride), base(base) {}

    // Create a view that points to the given base (contiguous stride, row-major)
    BhArray(const std::shared_ptr<BhBase<T>>& base, const Shape& shape,
            const size_t offset = 0)
          : offset(offset), shape(shape), stride(contiguous_stride(shape)), base(base) {}

    // Pretty printing the content of the array
    // TODO: for now it always print the flatten array
    void pprint(std::ostream& os) const;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const BhArray<T>& ary) {
    ary.pprint(os);
    return os;
}

}  // namespace bhxx

#endif