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
#include "SVector.hpp"
#include <ostream>
#include <vector>

namespace bhxx {

/** Class which enqueues an BhBase object for deletion
 *  with the Bohrium Runtime, but does not actually delete
 *  it straight away.
 *
 *  \note This is needed to ensure that all BhBase objects
 *  are still around until the list of instructions has
 *  been emptied.
 */
struct RuntimeDeleter {
    void operator()(BhBase* ptr) const;
};

/** Helper function to make shared pointers to BhBase,
 *  which use the RuntimeDeleter as their deleter */
template <typename... Args>
std::shared_ptr<BhBase> make_base_ptr(Args... args) {
    return std::shared_ptr<BhBase>(new BhBase(std::forward<Args>(args)...),
                                   RuntimeDeleter{});
}

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

    /** Create a new view */
    BhArray(Shape shape_, Stride stride_, const size_t offset_ = 0)
          : offset(offset_),
            shape(shape_),
            stride(std::move(stride_)),
            base(make_base_ptr(T(0), shape_.prod())) {
        assert(shape.size() == stride.size());
    }

    /** Create a new view (contiguous stride, row-major) */
    BhArray(Shape shape) : BhArray(shape, contiguous_stride(shape), 0) {}

    /** Create a view that points to the given base
     *
     *  \note The caller should make sure that the shared pointer
     *        uses the RuntimeDeleter as its deleter, since this is
     *        implicitly assumed throughout, i.e. if one wants to
     *        construct a BhBase object, use the make_base_ptr
     *        helper function.
     */
    BhArray(std::shared_ptr<BhBase> base_, Shape shape_, Stride stride_,
            const size_t offset_ = 0)
          : offset(offset_),
            shape(std::move(shape_)),
            stride(std::move(stride_)),
            base(std::move(base_)) {
        assert(shape.size() == stride.size());
    }

    /** Create a view that points to the given base (contiguous stride, row-major)
     *
     *  \note The caller should make sure that the shared pointer
     *        uses the RuntimeDeleter as its deleter, since this is
     *        implicitly assumed throughout, i.e. if one wants to
     *        construct a BhBase object, use the make_base_ptr
     *        helper function.
     */
    BhArray(std::shared_ptr<BhBase> base, Shape shape)
          : BhArray(std::move(base), shape, contiguous_stride(shape), 0) {
        assert(n_elem() == shape.prod());
    }

    //
    // Information
    //

    /** Return the rank of the BhArray */
    size_t rank() const {
        assert(shape.size() == stride.size());
        return shape.size();
    }

    /** Return the number of elements */
    size_t n_elem() const { return shape.prod(); }

    /** Return whether the view is contiguous and row-major */
    bool is_contiguous() const;

    //
    // Data access
    //
    /** Is the data referenced by this view's base array already
     *  allocated, i.e. initialised */
    bool is_data_initialised() const { return base->data != nullptr; }

    //@{
    /** Obtain the data pointer of the base array, not taking
     *  ownership of any kind.
     *
     *  \note This pointer might be a nullptr if the data in
     *        the base data is not initialised.
     *
     *  \note No flush is done automatically. The data might be
     *        out of sync with Bohrium.
     */
    const T* data() const { return static_cast<T*>(base->data); }
    T*       data() { return static_cast<T*>(base->data); }
    //@}

    //
    // Routines
    //

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
