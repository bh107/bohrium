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

#include <type_traits>
#include <ostream>
#include <bh_static_vector.hpp>
#include <bhxx/BhBase.hpp>
#include <bhxx/type_traits_util.hpp>
#include <bhxx/array_operations.hpp>

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
    void operator()(BhBase *ptr) const;
};

/** Helper function to make shared pointers to BhBase,
 *  which use the RuntimeDeleter as their deleter */
template<typename... Args>
std::shared_ptr<BhBase> make_base_ptr(Args... args) {
    return std::shared_ptr<BhBase>(new BhBase(std::forward<Args>(args)...), RuntimeDeleter{});
}

/** Static allocated shapes and strides that is interchangeable with standard C++ vector as long
 *  as the vector is smaller than `BH_MAXDIM`.
 */
typedef BhStaticVector<uint64_t> Shape;
using Stride = BhIntVec;

/** Return a contiguous stride (row-major) based on `shape` */
extern inline Stride contiguous_stride(const Shape &shape) {
    Stride ret(shape.size());
    int64_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        ret[i] = stride;
        stride *= shape[i];
    }
    return ret;
}

/** Core class that represent the core attributes of a view that isn't typed by its dtype */
class BhArrayUnTypedCore {
public:
    /// The array offset (from the start of the base in number of elements)
    uint64_t offset;
    /// The array shape (size of each dimension in number of elements)
    Shape shape;
    /// The array stride (the absolute stride of each dimension in number of elements)
    Stride stride;
    /// Pointer to the base of this array
    std::shared_ptr<BhBase> base;
    /// Metadata to support sliding views
    bh_slide slides;

    /** Return a `bh_view` of the array */
    bh_view getBhView() const {
        bh_view view;
        assert(base.use_count() > 0);
        view.base = base.get();
        view.start = static_cast<int64_t>(offset);
        if (shape.empty()) { // Scalar views (i.e. 0-dim views) are represented as 1-dim arrays with size one.
            view.ndim = 1;
            view.shape = BhIntVec({1});
            view.stride = BhIntVec({1});
        } else {
            view.ndim = static_cast<int64_t>(shape.size());
            view.shape = BhIntVec(shape.begin(), shape.end());
            view.stride = BhIntVec(stride.begin(), stride.end());;
        }
        view.slides = slides;
        return view;
    }
};

/** Representation of a multidimensional array that point to a `BhBase` array.
 *
 * @tparam T  The data type of the array and the underlying base array
 */
template<typename T>
class BhArray : public BhArrayUnTypedCore {
public:
    /// The data type of each array element
    typedef T scalar_type;

    /** Default constructor that leave the instance completely uninitialized. */
    BhArray() = default;

    /** Create a new view */
    explicit BhArray(Shape shape, Stride stride, uint64_t offset = 0) :
            BhArrayUnTypedCore{offset, std::move(shape), std::move(stride), make_base_ptr(T(0), shape.prod())} {
        assert(shape.size() == stride.size());
        assert(shape.prod() > 0);
    }

    /** Create a new view (contiguous stride, row-major) */
    explicit BhArray(Shape shape) : BhArray(std::move(shape), contiguous_stride(shape), 0) {}

    /** Create a view that points to the given base
     *
     *  \note The caller should make sure that the shared pointer
     *        uses the RuntimeDeleter as its deleter, since this is
     *        implicitly assumed throughout, i.e. if one wants to
     *        construct a BhBase object, use the make_base_ptr
     *        helper function.
     */
    explicit BhArray(std::shared_ptr<BhBase> base, Shape shape, Stride stride, uint64_t offset = 0) :
            BhArrayUnTypedCore{offset, std::move(shape), std::move(stride), std::move(base)} {
        assert(shape.size() == stride.size());
        assert(shape.prod() > 0);
    }

    /** Create a view that points to the given base (contiguous stride, row-major)
     *
     *  \note The caller should make sure that the shared pointer
     *        uses the RuntimeDeleter as its deleter, since this is
     *        implicitly assumed throughout, i.e. if one wants to
     *        construct a BhBase object, use the make_base_ptr
     *        helper function.
     */
    explicit BhArray(std::shared_ptr<BhBase> base, Shape shape) : BhArray(std::move(base), std::move(shape),
                                                                          contiguous_stride(shape), 0) {
        assert(static_cast<uint64_t>(base->nelem()) == shape.prod());
    }

    /** Create a copy of `ary` using a Bohrium `identity` operation.
     *
     *  \note This function implements implicit type conversion for all widening type casts
     */
    template<typename InType,
            typename std::enable_if<type_traits::is_safe_numeric_cast<scalar_type, InType>::value, int>::type = 0>
    BhArray(const BhArray<InType> &ary) {
        bhxx::identity(*this, ary);
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
    uint64_t size() const {
        return shape.prod();
    }

    /** Return whether the view is contiguous and row-major */
    bool isContiguous() const;

    //
    // Data access
    //
    /** Is the data referenced by this view's base array already
     *  allocated, i.e. initialised */
    bool isDataInitialised() const { return base->getDataPtr() != nullptr; }

    /** Obtain the data pointer of the array, not taking ownership of any kind.
     *
     *  \note This pointer might be a nullptr if the data in
     *        the base data is not initialised.
     *
     *  \note No flush is done automatically. The data might be
     *        out of sync with Bohrium.
     */
    T *data() {
        T *ret = static_cast<T *>(base->getDataPtr());
        if (ret == nullptr) {
            return nullptr;
        } else {
            return offset + ret;
        }
    }

    /// The const version of `data()`
    const T *data() const {
        const T *ret = static_cast<T *>(base->getDataPtr());
        if (ret == nullptr) {
            return nullptr;
        } else {
            return offset + ret;
        }
    }

    /// Pretty printing the content of the array
    void pprint(std::ostream &os) const;

    /// Returns a new view of the `idx` dimension. Negative index counts from the back
    BhArray<T> operator[](int64_t idx) const;
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const BhArray<T> &ary) {
    ary.pprint(os);
    return os;
}

}  // namespace bhxx
