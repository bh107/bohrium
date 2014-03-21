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
#include <iostream>

#include "traits.hpp"           // Traits for assigning type to constants and multi_arrays.
#include "runtime.hpp"
#include <sstream>

namespace bh {

//
//
//  CONSTRUCTORS
//
//
template <typename T>           // Default constructor - rank 0
multi_array<T>::multi_array() : temp(false), base(NULL)
{
    meta.base       = NULL;
    meta.ndim       = 0;
    meta.start      = 0;
    meta.shape[0]   = 0;
    meta.stride[0]  = 0;
}

template <typename T>           // Plain shaped constructor
multi_array<T>::multi_array(const uint64_t rank, const int64_t* sizes) : temp(false), base(NULL)
{
    meta.base       = NULL;
    meta.ndim       = rank;
    meta.start      = 0;

    int64_t stride = 1;                 // Setup strides
    for(int64_t i=meta.ndim-1; 0 <= i; --i) {
        meta.shape[i] = sizes[i];
        meta.stride[i] = stride;
        stride *= meta.shape[i];
    }
}

template <typename T>           // base/view constructor
multi_array<T>::multi_array(bh_base* _base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride) : temp(false), base(NULL)
{
    meta.ndim   = rank;
    meta.start  = start;
    meta.base   = _base;

    for(uint64_t i=0; i < rank; i++) {
        meta.shape[i]   = shape[i];
        meta.stride[i]  = stride[i];
    }
}

template <typename T>           // Copy constructor
multi_array<T>::multi_array(const multi_array<T>& operand) : temp(false), base(NULL)
{
    meta = operand.meta;
    meta.base = NULL;
    meta.start = 0;

    int64_t stride = 1;                 // Reset strides
    for(int64_t i=meta.ndim-1; 0 <= i; --i) {
        meta.stride[i] = stride;
        stride *= meta.shape[i];
    }
}

template <typename T>           // Copy constructor
template <typename OtherT>
multi_array<T>::multi_array(const multi_array<OtherT>& operand) : temp(false), base(NULL)
{
    meta.base   = NULL;
    meta.ndim   = operand.meta.ndim;
    meta.start  = 0;

    memcpy(meta.shape, operand.meta.shape, sizeof(bh_index)*BH_MAXDIM);

    int64_t stride = 1;                 // Reset strides
    for(int64_t i=meta.ndim-1; 0 <= i; --i) {
        meta.stride[i] = stride;
        stride *= meta.shape[i];
    }
}

template <typename T>
template <typename ...Dimensions>       // Variadic constructor
multi_array<T>::multi_array(Dimensions... shape) : temp(false), base(NULL)
{
    meta.base   = NULL;
    meta.ndim   = sizeof...(Dimensions);
    meta.start  = 0;

    unpack_shape(meta.shape, 0, shape...);

    int64_t stride = 1;                 // Setup strides
    for(int64_t i=meta.ndim-1; 0 <= i; --i) {
        meta.stride[i] = stride;
        stride *= meta.shape[i];
    }
}

template <typename T>                   // Deconstructor
multi_array<T>::~multi_array()
{
    if (base) {
        Runtime::instance().enqueue((bh_opcode)BH_FREE, *this);
        Runtime::instance().enqueue((bh_opcode)BH_DISCARD, *this);
        Runtime::instance().trash(base);
    }
}

//
// Methods
//

template <typename T>
inline
bh_base* multi_array<T>::getBase() const
{
    return base;
}

template <typename T>
inline
unsigned long multi_array<T>::getRank() const
{
    return (unsigned long)meta.ndim;
}

template <typename T>
inline
size_t multi_array<T>::len()
{
    size_t nelements = 1;
    for (int i = 0; i < meta.ndim; ++i) {
        nelements *= meta.shape[i];
    }
    return nelements;
}

template <typename T>
inline
int64_t multi_array<T>::shape(int64_t dim)
{
    if (dim>=meta.ndim) {
        throw std::runtime_error("Dude you are like totally out of bounds!\n");
    }

    return meta.shape[dim];
}

template <typename T>
void multi_array<T>::sync()
{
    Runtime::instance().enqueue((bh_opcode)BH_SYNC, *this);
    Runtime::instance().flush();
}

template <typename T>
typename multi_array<T>::iterator multi_array<T>::begin()
{
    this->sync();
    return multi_array<T>::iterator(meta);
}

template <typename T>
typename multi_array<T>::iterator multi_array<T>::end()
{
    return multi_array<T>::iterator();
}

//
// Increment / decrement
//
template <typename T>
multi_array<T>& multi_array<T>::operator++()
{
    if (!this->initialized()) {
        throw std::runtime_error("Err: Increment of a unintialized operand.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator++(int)
{
    if (!this->initialized()) {
        throw std::runtime_error("Err: Increment of a unintialized operand.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator--()
{
    if (!this->initialized()) {
        throw std::runtime_error("Err: Decrement of a unintialized operand.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator--(int)
{
    if (!this->initialized()) {
        throw std::runtime_error("Err: Decrement of a unintialized operand.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *this, *this, (T)1);
    return *this;
}

//
// Output / Printing
//
template <typename T>
std::ostream& operator<< (std::ostream& stream, multi_array<T>& rhs)
{
    if (!rhs.initialized()) {
        throw std::runtime_error("Err: Cannot output an unintialized operand.");
    }

    bool first = true;
    typename multi_array<T>::iterator it  = rhs.begin();
    typename multi_array<T>::iterator end = rhs.end();
    stream << "[ ";
    for(; it != end; it++) {
        if (!first) {
            stream  << ", ";
        } else {
            first = false;
        }
        stream << *it;
    }
    stream << " ]" << std::endl;

    if (rhs.getTemp()) {    // Cleanup temporary
        delete &rhs;
    }

    return stream;
}

//
// Slicing
//
template <typename T>
slice<T>& multi_array<T>::operator[](int rhs) {
    if (!initialized()) {
        throw std::runtime_error("Err: cannot slice an uninitialized operand.");
    }
    return (*(new slice<T>(*this)))[rhs];
}

template <typename T>
slice<T>& multi_array<T>::operator[](slice_range& rhs) {
    if (!initialized()) {
        throw std::runtime_error("Err: cannot slice an uninitialized operand.");
    }
    return (*(new slice<T>(*this)))[rhs];
}

//
// MISC
//

// Filling / assignment.
template <typename T>
multi_array<T>& multi_array<T>::operator=(const T& rhs)
{
    if (!this->initialized()) {
        if (this->meta.ndim != 0) {   // Very special case!
            // The operand has a shape but it is not initialized.
            // it based on its shape and assign the constant value
            link();
        } else {
            throw std::runtime_error("Err: cannot assign to an uninitialized operand.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *this, rhs);
    return *this;
}

template <typename T>
inline
bool multi_array<T>::getTemp() const
{
    return temp;
}

template <typename T>
inline
void multi_array<T>::setTemp(bool temp)
{
    this->temp = temp;
}

template <typename T>
inline
bool multi_array<T>::linked() const
{
    //return (key != 0);
    return (base != NULL);
}

template <typename T>
inline
bool multi_array<T>::initialized() const
{
    return (meta.base != NULL);
}

template <typename T>
inline
bool multi_array<T>::allocated() const
{
    return ((meta.base != NULL) && (meta.base->data != NULL));
}

// Linking - Assign a base to the multi_array.
template <typename T>
void multi_array<T>::link()
{
    if (base) {
        throw std::runtime_error("Dude you are ALREADY linked!");
    }
    base = new bh_base;
    assign_array_type<T>(base);
    base->nelem = bh_nelements(meta.ndim, meta.shape);
    base->data  = NULL;
    meta.base   = base;
}

template <typename T>
void multi_array<T>::link(bh_base *base_ptr)
{
    if (base) {
        throw std::runtime_error("Dude you are ALREADY linked!");
    }
    base      = base_ptr;
    meta.base = base;
}

template <typename T>
bh_base* multi_array<T>::unlink()
{
    if (!base) {
        throw std::runtime_error("Err: Unlinking operand which is not linked!");
    }

    bh_base *ret_base;
    ret_base = base;

    base      = NULL;
    meta.base = NULL;
    return ret_base;
}

//
//  Aliasing
//
template <typename T>
multi_array<T>& multi_array<T>::operator=(multi_array<T>& rhs)
{
    if ((base) && (base == rhs.getBase())) {  // Self-aliasing is a NOOP
        return *this;
    }

    if (base) {
        Runtime::instance().enqueue((bh_opcode)BH_FREE, *this);
        Runtime::instance().enqueue((bh_opcode)BH_DISCARD, *this);
        Runtime::instance().trash(base);
        unlink();
    }
                                // Create alias of rhs
    meta = rhs.meta;            // Inherit all meta

    if (rhs.getTemp()) {        // Take over temporary reference
        if (rhs.linked()) {
            link(rhs.unlink());
        }
        delete &rhs;            // Cleanup
    }

    return *this;
}

/**
 *  Aliasing via slicing
 *
 *  Construct a view based on a slice.
 *  Such as:
 *
 *  center = grid[_(1,-1,1)][_(1,-1,1)];
 */
template <typename T>
multi_array<T>& multi_array<T>::operator=(slice<T>& rhs)
{
    if (base) {
        Runtime::instance().enqueue((bh_opcode)BH_FREE, *this);
        Runtime::instance().enqueue((bh_opcode)BH_DISCARD, *this);
        Runtime::instance().trash(base);
        unlink();
    }

    multi_array<T>* vv = &rhs.view();
    this->meta = vv->meta;
    delete vv;

    return *this;
}

#ifndef NO_VARIADICS
/**
 *  Aliasing through reshaping.
 */
template <typename T, typename ...Dimensions>
multi_array<T>& view_as(multi_array<T>& rhs, Dimensions... shape)
{
    int64_t dims    = sizeof...(Dimensions),
            stride  = 1;

    if (!rhs.initialized()) {            // We do not have anything to view!
        throw std::runtime_error("Err: Trying to create a view "
                                 "of something that does not exist!\n");
    }

    multi_array<T>* result = &Runtime::instance().temp_view(rhs);
    unpack_shape(result->meta.shape, 0, shape...);
    result->meta.ndim = dims;

    for(int64_t i=dims-1; 0 <= i; --i) {        // Fix the stride
        result->meta.stride[i] = stride;
        stride *= result->meta.shape[i];
    }

    // TODO: Verify that the number of elements match

    return *result;
}
#endif

//
// Update
//
template <typename T>
multi_array<T>& multi_array<T>::operator()(multi_array& rhs)
{
    if (!initialized()) {   // We do not have anything to update!
        throw std::runtime_error("Err: You are trying to update "
                                 "something that does not exist!");
    }
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *this, rhs);

    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator()(const T& value) {

    if (!initialized()) {    // We do not have anything to update!
        throw std::runtime_error("Err: You are trying to update "
                                 "something that does not exist!");
    }
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *this, value);

    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator()(const void* data) {

    // We know nothing about the shape and size of this array
    if (!initialized()) {    
        throw std::runtime_error("Err: You are trying to update "
                                 "something that does not exist!");
    }

    size_t nbytes = (this->base->nelem) * bh_type_size(this->base->type);

    // Ensure that we have memory to write to.
    if (!allocated()) {
        this->base->data = bh_memory_malloc(nbytes);
    }

    // Copy the data
    memcpy(this->base->data, data, nbytes);

    return *this;
}

//
// Typecasting
//
template <typename T, typename FromT>
multi_array<T>& as(multi_array<FromT>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);

    return *result;
}

// NON-MEMBER STUFF
template <typename T>
multi_array<T>& copy(multi_array<T>& rhs)
{
    if (!rhs.linked()) {
        throw std::runtime_error("Far out dude! you are trying create a copy "
                                 "of something that does not exist!\n");
    }

    multi_array<T>* result = &Runtime::instance().temp<T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);

    return *result;
}

template <typename T>
multi_array<T>& flatten(multi_array<T>& rhs)
{
    if (!rhs.linked()) {
        throw std::runtime_error("Far out dude! you are trying to flatten "
                                 "something that does not exist!\n");
    }

    throw std::runtime_error("flatten: Not implemented.\n");

    /*multi_array<T>* result = &Runtime::instance().temp<T>();
    result->meta.ndim = 1;
    result->meta.start = 0;
    result->meta.shape[rhs.len()];
    result->meta.stride[0] = 1;
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);

    return *result;*/
}

template <typename T>
multi_array<T>& transpose(multi_array<T>& rhs)
{
    if (!rhs.linked()) {
        throw std::runtime_error("Far out dude! you are trying to transpose "
                                 "something that does not exist!\n");
    }

    multi_array<T>* result = &Runtime::instance().temp_view<T>(rhs);

    for(bh_intp i=0, j=result->meta.ndim-1; i<result->meta.ndim; ++i, --j) {
        result->meta.stride[i]  = rhs.meta.stride[j];
        result->meta.shape[i]   = rhs.meta.shape[j];
    }

    return *result;
}

}

