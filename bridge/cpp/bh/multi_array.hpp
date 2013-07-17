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

template <typename T>
void multi_array<T>::init()     // Pseudo-default constructor
{                               // Associates a multi_array
                                // with a bh_base
    key     = Runtime::instance().keys++;
    temp    = false;
    Runtime::instance().storage.insert(key, new bh_base);
    assign_array_type<T>(&Runtime::instance().storage[key]);
}

template <typename T>           // Default constructor - rank 0
multi_array<T>::multi_array() : key(0), temp(false) { }

/**
 * Inherit meta-data (ndim, start, shape, stride) from another operand.
 * Does not copy data.
 *
 */
template <typename T>           // Copy constructor
multi_array<T>::multi_array(const multi_array<T>& operand)
{
    init();

    meta = operand.getMeta();

}

template <typename T>
template <typename ...Dimensions>       // Variadic constructor
multi_array<T>::multi_array(Dimensions... shape)
{
    init();

    meta.base        = NULL;
    meta.ndim        = sizeof...(Dimensions);
    meta.start       = 0;

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
    if (key>0) {
        if (NULL == meta.base) {        // Only send free on base-array
            Runtime::instance().enqueue((bh_opcode)BH_FREE, *this);
        }
        Runtime::instance().enqueue((bh_opcode)BH_DISCARD, *this);
        Runtime::instance().trash(key);
    }
}

template <typename T>
inline
bh_view multi_array<T>::getMeta() const
{
    return meta;
}

template <typename T>
inline
size_t multi_array<T>::getKey() const
{
    return key;
}

template <typename T>
inline
unsigned long multi_array<T>::getRank() const
{
    return (unsigned long)meta.ndim;
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
size_t multi_array<T>::len()
{
    size_t nelements = 1;
    for (int i = 0; i < Runtime::instance().storage[key].ndim; ++i) {
        nelements *= Runtime::instance().storage[key].shape[i];
    }
    return nelements;
}

template <typename T>
inline
int64_t multi_array<T>::shape(int64_t dim)
{
    if (dim>=Runtime::instance().storage[key].ndim) {
        throw std::runtime_error("Dude you are like totally out of bounds!\n");
    }

    return Runtime::instance().storage[key].shape[dim];
}

template <typename T>
typename multi_array<T>::iterator multi_array<T>::begin()
{
    Runtime::instance().enqueue((bh_opcode)BH_SYNC, *this);
    Runtime::instance().flush();

    return multi_array<T>::iterator(Runtime::instance().storage[this->key], meta);
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
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator++(int)
{
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator--()
{
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator--(int)
{
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *this, *this, (T)1);
    return *this;
}

//
// Output / Printing
//
template <typename T>
std::ostream& operator<< (std::ostream& stream, multi_array<T>& rhs)
{
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
    return (*(new slice<T>(*this)))[rhs];
}
                                                        
template <typename T>
slice<T>& multi_array<T>::operator[](slice_range& rhs) {
    return (*(new slice<T>(*this)))[rhs];
}



// Initialization
template <typename T>
multi_array<T>& multi_array<T>::operator=(const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *this, rhs);
    return *this;
}

// Linking
template <typename T>
void multi_array<T>::link(const size_t ext_key)
{
    if (0!=key) {
        throw std::runtime_error("Dude you are ALREADY linked!");
    }
    key = ext_key;
}

template <typename T>
size_t multi_array<T>::unlink()
{
    if (0==key) {
        throw std::runtime_error("Dude! THis one aint linked at all!");
    }

    size_t retKey = key;
    key = 0;
    return retKey;
}

// Aliasing
template <typename T>
multi_array<T>& multi_array<T>::operator=(multi_array<T>& rhs)
{
    if (key != rhs.getKey()) {      // Prevent self-aliasing
        
        if (key>0) {                // Release current linkage
            if (NULL == meta.base) {
                Runtime::instance().enqueue((bh_opcode)BH_FREE, *this);
            }
            Runtime::instance().enqueue((bh_opcode)BH_DISCARD, *this);
            unlink();
        }

        if (rhs.getTemp()) {        // Take over temporary reference
            link(rhs.unlink());
            temp = false;
            delete &rhs;
        } else {                    // Create an alias of rhs.
            init();

            meta.base   = rhs.meta.base;
            meta.ndim   = rhs.meta.ndim;
            meta.start  = rhs.meta.start;
            for(int64_t i=0; i< rhs.meta.ndim; i++) {
                meta.shape[i] = rhs.meta.shape[i];
            }
            for(int64_t i=0; i< rhs.meta.ndim; i++) {
                meta.stride[i] = rhs.meta.stride[i];
            }
        }
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
 *
 *  TODO: this is probobaly not entirely correct...
 */
template <typename T>
multi_array<T>& multi_array<T>::operator=(slice<T>& rhs)
{
    multi_array<T>* vv = &rhs.view();

    if (key>0) {                // Release current linkage
        if (NULL == Runtime::instance().storage[key].base) {
            Runtime::instance().enqueue((bh_opcode)BH_FREE, *this);
        }
        Runtime::instance().enqueue((bh_opcode)BH_DISCARD, *this);
        unlink();
    }

    link(vv->unlink());
    delete vv;

    return *this;
}




//
// Update
//
template <typename T>
multi_array<T>& multi_array<T>::operator()(multi_array& rhs)
{
    if (1>key) {    // We do not have anything to update!
        throw std::runtime_error("Far out dude! you are trying to update "
                                 "something that does not exist!");
    }
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *this, rhs);

    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator()(const T& value) {

    if (1>key) {    // We do not have anything to update!
        throw std::runtime_error("Far out dude! you are trying to update "
                                 "something that does not exist!");
    }
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *this, value);

    return *this;
}

//
// Typecasting
//
template <typename T, typename FromT>
multi_array<T>& as(multi_array<FromT>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T>();
    result->setTemp(true);

    Runtime::instance().storage[result->getKey()].base        = NULL;
    Runtime::instance().storage[result->getKey()].ndim        = Runtime::instance().storage[rhs.getKey()].ndim;
    Runtime::instance().storage[result->getKey()].start       = Runtime::instance().storage[rhs.getKey()].start;
    for(int64_t i=0; i< Runtime::instance().storage[rhs.getKey()].ndim; i++) {
        Runtime::instance().storage[result->getKey()].shape[i] = Runtime::instance().storage[rhs.getKey()].shape[i];
    }
    for(int64_t i=0; i< Runtime::instance().storage[rhs.getKey()].ndim; i++) {

        Runtime::instance().storage[result->getKey()].stride[i] = Runtime::instance().storage[rhs.getKey()].stride[i];
    }
    Runtime::instance().storage[result->getKey()].data        = NULL;

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);

    return *result;
}

// Create a view of an operand which has a different shape
template <typename T, typename ...Dimensions>
multi_array<T>& view_as(multi_array<T>& rhs, Dimensions... shape)
{
    int64_t dims    = sizeof...(Dimensions),
            stride  = 1;

    if (1>rhs.getKey()) {            // We do not have anything to view!
        throw std::runtime_error("Far out dude! you are trying create a view "
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

// NON-MEMBER STUFF
template <typename T>
multi_array<T>& copy(multi_array<T>& rhs)
{
    if (1>rhs.getKey()) {   // We do not have anything to copy!
        throw std::runtime_error("Far out dude! you are trying create a copy "
                                 "of something that does not exist!\n");
    }

    multi_array<T>* result = &Runtime::instance().temp<T>();
    result->setTemp(true);

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);

    return *result;
}

template <typename T>
multi_array<T>& flatten(multi_array<T>& rhs)
{
    if (1>rhs.getKey()) {   // We do not have anything to copy!
        throw std::runtime_error("Far out dude! you are trying to flatten "
                                 "something that does not exist!\n");
    }

    throw std::runtime_error("flatten: Not implemented.\n");

    multi_array<T>* result = &Runtime::instance().temp<T>();
    result->setTemp(true);

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);

    return *result;
}

template <typename T>
multi_array<T>& transpose(multi_array<T>& rhs)
{
    if (1>rhs.getKey()) {   // We do not have anything to copy!
        throw std::runtime_error("Far out dude! you are trying to transpose "
                                 "something that does not exist!\n");
    }

    throw std::runtime_error("transpose: Not implemented.\n");

    multi_array<T>* result = &Runtime::instance().temp<T>();
    result->setTemp(true);

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);

    return *result;
}

}

