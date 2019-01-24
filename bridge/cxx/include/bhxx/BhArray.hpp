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
protected:
    /// The array offset (from the start of the base in number of elements)
    uint64_t _offset = 0;
    /// The array shape (size of each dimension in number of elements)
    Shape _shape;
    /// The array stride (the absolute stride of each dimension in number of elements)
    Stride _stride;
    /// Pointer to the base of this array
    std::shared_ptr<BhBase> _base;
    /// Metadata to support sliding views
    bh_slide _slides;
public:

    /** Default constructor that leave the instance completely uninitialized */
    BhArrayUnTypedCore() = default;

    /** Constructor to initiate all but the `_slides` attribute */
    BhArrayUnTypedCore(uint64_t offset, Shape shape, Stride stride, std::shared_ptr<BhBase> base) :
            _offset(offset), _shape(std::move(shape)), _stride(std::move(stride)), _base(std::move(base)) {
        if (_shape.size() != _stride.size()) {
            throw std::runtime_error("The shape and stride must have same length");
        }
        if (shape.prod() <= 0) {
            throw std::runtime_error("The total size must be greater than zero");
        }
    }

    /** Return a `bh_view` of the array */
    bh_view getBhView() const {
        bh_view view;
        assert(_base);
        view.base = _base.get();
        view.start = static_cast<int64_t>(offset());
        if (shape().empty()) { // Scalar views (i.e. 0-dim views) are represented as 1-dim arrays with size one.
            view.ndim = 1;
            view.shape = BhIntVec({1});
            view.stride = BhIntVec({1});
        } else {
            view.ndim = static_cast<int64_t>(shape().size());
            view.shape = BhIntVec(shape().begin(), shape().end());
            view.stride = BhIntVec(_stride.begin(), _stride.end());;
        }
        view.slides = _slides;
        return view;
    }

    /** Swapping `a` and `b` */
    friend void swap(BhArrayUnTypedCore &a, BhArrayUnTypedCore &b) noexcept {
        using std::swap; // enable ADL
        swap(a._offset, b._offset);
        swap(a._shape, b._shape);
        swap(a._stride, b._stride);
        swap(a._base, b._base);
        swap(a._slides, b._slides);
    }

    /** Return the offset of the array */
    uint64_t offset() const {
        return _offset;
    }

    /** Return the shape of the array */
    const Shape &shape() const {
        return _shape;
    }

    /** Return the stride of the array */
    const Stride &stride() const {
        return _stride;
    }

    /** Return the base of the array */
    const std::shared_ptr<BhBase> &base() const {
        return _base;
    }

    /** Return the base of the array */
    std::shared_ptr<BhBase> &base() {
        return _base;
    }

    /** Set the shape and stride of the array (both must have the same lenth) */
    void setShapeAndStride(Shape shape, Stride stride) {
        if (shape.size() != stride.size()) {
            throw std::runtime_error("The shape and stride must have same length");
        }
        _shape = std::move(shape);
        _stride = std::move(stride);
    }

    /** Return the slides object of the array */
    const bh_slide &slides() const {
        return _slides;
    }

    /** Return the slides object of the array */
    bh_slide &slides() {
        return _slides;
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

    /** Create a new array. `Shape` and `Stride` must have the same length.
     *
     * @param shape   Shape of the new array
     * @param stride  Stride of the new array
     */
    explicit BhArray(Shape shape, Stride stride) : BhArrayUnTypedCore{0, std::move(shape), std::move(stride),
                                                                      make_base_ptr(T(0), shape.prod())} {}

    /** Create a new array (contiguous stride, row-major) */
    explicit BhArray(Shape shape) : BhArray(std::move(shape), contiguous_stride(shape)) {}

    /** Create a array that points to the given base
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

    /** Create a copy of `ary` using a Bohrium `identity` operation, which copies the underlying array data.
     *
     *  \note This function implements implicit type conversion for all widening type casts
     */
    template<typename InType,
            typename std::enable_if<type_traits::is_safe_numeric_cast<scalar_type, InType>::value, int>::type = 0>
    BhArray(const BhArray<InType> &ary) : BhArray(ary.shape()) {
        bhxx::identity(*this, ary);
    }

    /** Copy constructor that only copies meta data. The underlying array data is untouched */
    BhArray(const BhArray &) = default;

    /** Move constructor that only moves meta data. The underlying array data is untouched */
    BhArray(BhArray &&) noexcept = default;

    /** Copy the data of `other` into the array using a Bohrium `identity` operation */
    BhArray<T> &operator=(const BhArray<T> &other) {
        bhxx::identity(*this, other);
        return *this;
    }

    /** Copy the data of `other` into the array using a Bohrium `identity` operation
     *
     *  \note A move assignment is the same as a copy assignment.
     */
    BhArray<T> &operator=(BhArray<T> &&other) {
        bhxx::identity(*this, other);
        other.reset();
        return *this;
    }

    /**  Copy the scalar of `scalar_value` into the array using a Bohrium `identity` operation */
    template<typename InType,
            typename std::enable_if<type_traits::is_arithmetic<InType>::value, int>::type = 0>
    BhArray<T> &operator=(const InType &scalar_value) {
        bhxx::identity(*this, scalar_value);
        return *this;
    }

    /** Reset the array to `ary` */
    void reset(BhArray<T> ary) noexcept {
        swap(*this, ary);
    }

    /** Reset the array by cleaning all meta data and leave the array uninitialized. */
    void reset() noexcept {
        reset(BhArray());
    }

    /** Return the rank of the BhArray */
    size_t rank() const {
        assert(shape().size() == _stride.size());
        return shape().size();
    }

    /** Return the number of elements */
    uint64_t size() const {
        return shape().prod();
    }

    /** Return whether the view is contiguous and row-major */
    bool isContiguous() const;

    /** Is the data referenced by this view's base array already
     *  allocated, i.e. initialised */
    bool isDataInitialised() const { return _base->getDataPtr() != nullptr; }

    /** Obtain the data pointer of the array, not taking ownership of any kind.
     *
     * @param flush  Should we flush the runtime system before retrieving the data pointer
     * @return       The data pointer that might be a nullptr if the data in
     *               the base data is not initialised.
     */
    const T *data(bool flush = true) const;

    /// The const version of `data()`
    T *data(bool flush = true) {
        const BhArray<T> *t = this;
        const T *ret = data(t);
        return const_cast<T *>(ret);
    }

    /// Pretty printing the content of the array
    void pprint(std::ostream &os) const;

    /// Returns a new view of the `idx` dimension. Negative index counts from the back
    BhArray<T> operator[](int64_t idx) const;

    /// Return a new transposed view
    BhArray<T> transpose() const;

    /// Return a new reshaped view (the array must be contiguous)
    BhArray<T> reshape(Shape shape) const;
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const BhArray<T> &ary) {
    ary.pprint(os);
    return os;
}

}  // namespace bhxx
