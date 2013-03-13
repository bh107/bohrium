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

//
//  WARN:   This file is generated; changes to it will be overwritten.
//          If you wish to change its functionality then change the code-generator for this file.
//          Take a look at: codegen/README
//

#ifndef __BOHRIUM_BRIDGE_CPP_OPERATORS
#define __BOHRIUM_BRIDGE_CPP_OPERATORS
#include "bh.h"

namespace bh {
//
//  Internally defined operator overloads
//

template <typename T>
multi_array<T>& multi_array<T>::operator+= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator+= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;

        // Create view pointing to rhs as base
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator-= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator-= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;

        // Create view pointing to rhs as base
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator*= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator*= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;

        // Create view pointing to rhs as base
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator/= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator/= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;

        // Create view pointing to rhs as base
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator%= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator%= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;

        // Create view pointing to rhs as base
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator&= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator&= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;

        // Create view pointing to rhs as base
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator|= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator|= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;

        // Create view pointing to rhs as base
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator^= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator^= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;

        // Create view pointing to rhs as base
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_IDENTITY, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
    if (!compatible_shape(*this, *input)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to rhs
        input = &Runtime::instance()->view(rhs);
        // Broadcast the view
    }
    Runtime::instance()->enqueue((bh_opcode)BH_IDENTITY, *this, *input);
    return *this;
}

//
//  Binary operators such as:
//  Mapping "a + b" to BH_ADD(t, a, b)
//  Mapping "a + 1.0" to BH_ADD(t, a, 1.0)
//  Mapping "1.0 + a" to BH_ADD(t, 1.0, a)
//

template <typename T>
multi_array<T>& operator+ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator+ (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator+ (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator- (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator- (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator- (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator* (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator* (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator* (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator/ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator/ (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator/ (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator% (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator% (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator% (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator== (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator== (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator== (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator!= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator!= (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator!= (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator> (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator> (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator> (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator>= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator>= (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator>= (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator< (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator< (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator< (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator<= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator<= (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator<= (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator&& (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator&& (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator&& (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator|| (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator|| (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator|| (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator& (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator& (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator& (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator| (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator| (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator| (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator^ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    if (!compatible_shape(lhs, rhs)) {
        std::cout << "Incompatible shape, possibly broadcastable." << std::endl;
        // Create a view pointing to the smallest operand.
        // Broadcast the shape and stride for the view
    }

    // Shape of "result" should be defined by broadcast
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator^ (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator^ (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

//
//  Unary operators such as:
//  Mapping "!a" to BH_NEGATE(t, a)
//

template <typename T>
multi_array<T> & operator! (multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_NOT, *result, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator~ (multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_INVERT, *result, rhs);

    return *result;
}

}
#endif
