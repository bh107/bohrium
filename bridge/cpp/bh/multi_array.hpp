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
#include "cppb.hpp"
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

template <typename T>           // Default constructor
multi_array<T>::multi_array()
{
    init(); //rank = 0;

    DEBUG_PRINT(">> array(): key(%d)\n", key);
}

template <typename T>           // Copy constructor
multi_array<T>::multi_array(const multi_array<T>& operand)
{
    init(); //rank = operand.getRank();

    storage[key].base        = NULL;
    storage[key].ndim        = storage[operand.getKey()].ndim;
    storage[key].start       = storage[operand.getKey()].start;
    for(bh_index i=0; i< storage[operand.getKey()].ndim; i++) {
        storage[key].shape[i] = storage[operand.getKey()].shape[i];
    }
    for(bh_index i=0; i< storage[operand.getKey()].ndim; i++) {
        storage[key].stride[i] = storage[operand.getKey()].stride[i];
    }
    storage[key].data        = NULL;

    DEBUG_PRINT(">> array(op): key(%d)\n", key);
}

template <typename T>       // "Vector-like" constructor
multi_array<T>::multi_array(unsigned int n)
{
    init(); //rank = 1;

    storage[key].base        = NULL;
    storage[key].ndim        = 1;
    storage[key].start       = 0;
    storage[key].shape[0]    = n;
    storage[key].stride[0]   = 1;
    storage[key].data        = NULL;

    DEBUG_PRINT(">> array(int): key(%d)\n", key);
}


template <typename T>       // "Matrix-like" constructor
multi_array<T>::multi_array(unsigned int m, unsigned int n)
{
    init(); //rank = 2;

    storage[key].base        = NULL;
    storage[key].ndim        = 2;
    storage[key].start       = 0;
    storage[key].shape[0]    = m;
    storage[key].stride[0]   = n;
    storage[key].shape[1]    = n;
    storage[key].stride[1]   = 1;
    storage[key].data        = NULL;

    DEBUG_PRINT(">> array(int, int): key(%d)\n", key );
}

template <typename T>       // "Matrix-like" constructor
multi_array<T>::multi_array(unsigned int d2, unsigned int d1, unsigned int d0)
{
    init(); //rank = 3;

    storage[key].base        = NULL;
    storage[key].ndim        = 3;
    storage[key].start       = 0;
    storage[key].shape[0]    = d2;
    storage[key].stride[0]   = d1;
    storage[key].shape[1]    = d1;
    storage[key].stride[1]   = d0;
    storage[key].shape[2]    = d0;
    storage[key].stride[2]   = 1;
    storage[key].data        = NULL;

    DEBUG_PRINT(">> array(int, int, int): key(%d)\n", key);
}


template <typename T>       // Deconstructor
multi_array<T>::~multi_array()
{
    DEBUG_PRINT("> Deconstructing(): key(%d)\n", key);

    Runtime::instance()->enqueue((bh_opcode)BH_FREE, *this);
    Runtime::instance()->enqueue((bh_opcode)BH_DISCARD, *this);
}

template <typename T>
inline
unsigned int multi_array<T>::getKey() const
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

template <typename T>
std::ostream& operator<< (std::ostream& stream, multi_array<T>& rhs)
{
    bool first = true;
    multi_array<double>::iterator it  = rhs.begin();
    multi_array<double>::iterator end = rhs.end();

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

    return stream;
}

}

