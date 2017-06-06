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

#pragma once

#include "BhBase.hpp"
#include "util.hpp"
#include <ostream>
#include <vector>

namespace bhxx {

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
    std::shared_ptr<BhBase> base;

    // Create a new view that points to a new base
    BhArray(Shape shape, Stride stride, const size_t offset = 0)
          : offset(offset),
            shape(shape),
            stride(std::move(stride)),
            base(new BhBase(shape.prod()), BhBaseDeleter{}) {
        base->set_type<T>();
    }

    // Create a new view that points to a new base (contiguous stride, row-major)
    BhArray(Shape shape, const size_t offset = 0)
          : BhArray(shape, contiguous_stride(shape), offset) {}

    // Create a view that points to the given base
    BhArray(std::shared_ptr<BhBase> base, Shape shape, Stride stride,
            const size_t offset = 0)
          : offset(offset),
            shape(std::move(shape)),
            stride(std::move(stride)),
            base(std::move(base)) {}

    // Create a view that points to the given base (contiguous stride, row-major)
    BhArray(std::shared_ptr<BhBase> base, Shape shape, const size_t offset = 0)
          : BhArray(std::move(base), shape, contiguous_stride(shape), offset) {}

    // Pretty printing the content of the array
    // TODO: for now it always print the flatten array
    void pprint(std::ostream& os) const;

    //@{
    /** Obtain the data pointer of the base array, not taking
     *  ownership of any kind.
     *
     *  \note This pointer might be a nullptr if the data in
     *        the base is not yet initialised
     *
     * \note You can always force initialisation using
     *  ```
     *      sync(array);
     *      Runtime::instance().flush();
     *  ```
     *  after which ``data()`` should not be nullptr.
     */
    const T* data() const { return static_cast<T*>(base->data); }
    T*       data() { return static_cast<T*>(base->data); }
    //@}
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const BhArray<T>& ary) {
    ary.pprint(os);
    return os;
}

}  // namespace bhxx
