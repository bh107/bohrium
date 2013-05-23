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
{
    key     = keys++;
    temp    = false;
    storage.insert(key, new bh_array);
    assign_array_type<T>(&storage[key]);
}

template <typename T>           // Default constructor - rank 0
multi_array<T>::multi_array() : key(0), temp(false) { }

/**
 * Inherit ndim / shape from another operand
 * Does not copy data.
 *
 */
template <typename T>           // Copy constructor
multi_array<T>::multi_array(const multi_array<T>& operand)
{
    init();

    storage[key].data        = NULL;
    storage[key].base        = NULL;
    storage[key].ndim        = storage[operand.getKey()].ndim;
    storage[key].start       = storage[operand.getKey()].start;

    for(int64_t i=0; i< storage[operand.getKey()].ndim; i++) {
        storage[key].shape[i] = storage[operand.getKey()].shape[i];
    }
    for(int64_t i=0; i< storage[operand.getKey()].ndim; i++) {
        storage[key].stride[i] = storage[operand.getKey()].stride[i];
    }
}

template <typename T>
template <typename ...Dimensions>       // Variadic constructor
multi_array<T>::multi_array(Dimensions... shape)
{
    init();

    storage[key].data        = NULL;
    storage[key].base        = NULL;
    storage[key].ndim        = sizeof...(Dimensions);
    storage[key].start       = 0;

    unpack_shape(storage[key].shape, 0, shape...);

    int64_t stride = 1;                 // Setup strides
    for(int64_t i=storage[key].ndim-1; 0 <= i; --i) {
        storage[key].stride[i] = stride;
        stride *= storage[key].shape[i];
    }
}

template <typename T>                   // Deconstructor
multi_array<T>::~multi_array()
{
    if (key>0) {
        std::cout << "~multi_array() Addr: " << &storage[key] << std::endl;
        if (NULL == storage[key].base) {    // Only send free on base-array
            Runtime::instance()->enqueue((bh_opcode)BH_FREE, *this);
        }
        Runtime::instance()->enqueue((bh_opcode)BH_DISCARD, *this);
        Runtime::instance()->trash(key);
    }
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
    return (unsigned long)*(&storage[key].ndim);
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
    for (int i = 0; i < storage[key].ndim; ++i) {
        nelements *= storage[key].shape[i];
    }
    return nelements;
}

template <typename T>
typename multi_array<T>::iterator multi_array<T>::begin()
{
    Runtime::instance()->enqueue((bh_opcode)BH_SYNC, *this);
    Runtime::instance()->flush();

    return multi_array<T>::iterator(storage[this->key]);
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
    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator++(int)
{
    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator--()
{
    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *this, *this, (T)1);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator--(int)
{
    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *this, *this, (T)1);
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
        std::cout << "<< delete temp!" << std::endl;
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

//
// Reshaping
//
template <typename T>
multi_array<T>& multi_array<T>::operator()(const T& n) {
    std::cout << "Reshape to: " << n << std::endl;
    return *this;
}

// Initialization
template <typename T>
multi_array<T>& multi_array<T>::operator=(const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_IDENTITY, *this, rhs);
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
            if (NULL == storage[key].base) {
                Runtime::instance()->enqueue((bh_opcode)BH_FREE, *this);
            }
            Runtime::instance()->enqueue((bh_opcode)BH_DISCARD, *this);
            unlink();
        }

        if (rhs.getTemp()) {        // Take over temporary reference
            link(rhs.unlink());
            temp = false;
            delete &rhs;
        } else {                    // Create an alias of rhs.
            init();

            storage[key].data       = NULL;
            storage[key].base       = &storage[rhs.getKey()];
            storage[key].ndim       = storage[rhs.getKey()].ndim;
            storage[key].start      = storage[rhs.getKey()].start;
            for(int64_t i=0; i< storage[rhs.getKey()].ndim; i++) {
                storage[key].shape[i] = storage[rhs.getKey()].shape[i];
            }
            for(int64_t i=0; i< storage[rhs.getKey()].ndim; i++) {
                storage[key].stride[i] = storage[rhs.getKey()].stride[i];
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
        if (NULL == storage[key].base) {
            Runtime::instance()->enqueue((bh_opcode)BH_FREE, *this);
        }
        Runtime::instance()->enqueue((bh_opcode)BH_DISCARD, *this);
        unlink();
    }

    link(vv->unlink());
    delete vv;

    return *this;
}


//
// Typecasting
//
template <typename T> template <typename Ret>
multi_array<Ret>& multi_array<T>::as()
{
    multi_array<Ret>* result = &Runtime::instance()->temp<Ret>();
    result->setTemp(true);

    storage[result->getKey()].base        = NULL;
    storage[result->getKey()].ndim        = storage[this->getKey()].ndim;
    storage[result->getKey()].start       = storage[this->getKey()].start;
    for(int64_t i=0; i< storage[this->getKey()].ndim; i++) {
        storage[result->getKey()].shape[i] = storage[this->getKey()].shape[i];
    }
    for(int64_t i=0; i< storage[this->getKey()].ndim; i++) {

        storage[result->getKey()].stride[i] = storage[this->getKey()].stride[i];
    }
    storage[result->getKey()].data        = NULL;

    Runtime::instance()->enqueue((bh_opcode)BH_IDENTITY, *result, *this);

    return *result;
}

/**
 *  Update operand!
 */
template <typename T>
multi_array<T>& multi_array<T>::update(multi_array& rhs)
{
    if (1>key) {    // We do not have anything to update!
        throw std::runtime_error("Far out dude! you are trying to update "
                                 "something that does not exist!");
    }
    Runtime::instance()->enqueue((bh_opcode)BH_IDENTITY, *this, rhs);

    return *this;
}

}

