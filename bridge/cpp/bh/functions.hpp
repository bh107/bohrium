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

#ifndef __BOHRIUM_BRIDGE_CPP_FUNCTIONS
#define __BOHRIUM_BRIDGE_CPP_FUNCTIONS
#include "bh.h"

namespace bh {

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& add (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& add (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& add (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& add (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& add (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& add (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& subtract (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& subtract (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& subtract (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& subtract (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& subtract (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& subtract (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& multiply (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& multiply (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& multiply (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& multiply (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& multiply (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& multiply (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& divide (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& divide (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& divide (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& divide (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& divide (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& divide (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& mod (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& mod (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& mod (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& mod (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& mod (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& mod (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& bitwise_and (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bitwise_and (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bitwise_and (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& bitwise_and (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& bitwise_and (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& bitwise_and (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& bitwise_or (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bitwise_or (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bitwise_or (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& bitwise_or (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& bitwise_or (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& bitwise_or (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& bitwise_xor (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bitwise_xor (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bitwise_xor (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& bitwise_xor (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& bitwise_xor (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& bitwise_xor (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& left_shift (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& left_shift (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& left_shift (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& left_shift (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& left_shift (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& left_shift (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& right_shift (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& right_shift (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& right_shift (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& right_shift (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& right_shift (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& right_shift (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& equal (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& equal (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& equal (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& equal (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& equal (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& not_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& not_equal (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& not_equal (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& not_equal (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& not_equal (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& not_equal (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& greater (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& greater (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& greater (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& greater (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& greater (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& greater (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& greater_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& greater_equal (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& greater_equal (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& greater_equal (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& greater_equal (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& greater_equal (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& less (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& less (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& less (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& less (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& less (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& less (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& less_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& less_equal (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& less_equal (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& less_equal (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& less_equal (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& less_equal (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& logical_and (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& logical_and (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& logical_and (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& logical_and (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& logical_and (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& logical_and (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& logical_or (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& logical_or (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& logical_or (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& logical_or (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& logical_or (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& logical_or (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& logical_xor (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& logical_xor (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& logical_xor (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& logical_xor (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& logical_xor (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& logical_xor (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, rhs);
    return res;
}

// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& logical_not (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& logical_not (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& invert (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_INVERT, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& invert (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_INVERT, res, rhs);
    return res;
}
// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& pow (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& pow (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& pow (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& pow (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& pow (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& pow (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& power (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& power (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& power (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& power (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& power (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& power (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& abs (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& abs (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& absolute (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& absolute (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, res, rhs);
    return res;
}
// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& max (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& max (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& max (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& max (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& max (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& max (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& maximum (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& maximum (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& maximum (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& maximum (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& maximum (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& maximum (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& min (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& min (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& min (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& min (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& min (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& min (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& minimum (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& minimum (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& minimum (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& minimum (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& minimum (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& minimum (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& sin (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SIN, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& sin (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SIN, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& cos (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_COS, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& cos (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_COS, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& tan (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TAN, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& tan (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_TAN, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& sinh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SINH, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& sinh (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SINH, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& cosh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_COSH, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& cosh (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_COSH, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& tanh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TANH, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& tanh (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_TANH, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& asin (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& asin (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& acos (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& acos (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& atan (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& atan (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, res, rhs);
    return res;
}
// Implicit creation and use of temporary array for result
template <typename T>
multi_array<T>& atan2 (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& atan2 (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& atan2 (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, lhs, rhs);
    return *result;
}

// Explicit use of existing array for result
template <typename T>
multi_array<T>& atan2 (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& atan2 (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, lhs, rhs);
    return res;
}

template <typename T>
multi_array<T>& atan2 (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, lhs, rhs);
    return res;
}

// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& asinh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& asinh (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& acosh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& acosh (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& atanh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& atanh (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& exp (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXP, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& exp (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EXP, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& exp2 (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXP2, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& exp2 (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EXP2, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& expm1 (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& expm1 (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& isnan (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& isnan (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& isinf (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ISINF, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& isinf (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ISINF, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& log (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& log (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOG, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& log2 (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG2, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& log2 (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOG2, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& log10 (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG10, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& log10 (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOG10, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& log1p (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& log1p (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& sqrt (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SQRT, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& sqrt (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SQRT, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& ceil (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_CEIL, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& ceil (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_CEIL, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& trunc (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& trunc (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& floor (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& floor (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, res, rhs);
    return res;
}
// Implicit creation of temporary array for result
template <typename T>
multi_array<T>& rint (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RINT, *result, rhs);
    return *result;
}
// Explicit use of existing array for result
template <typename T>
multi_array<T>& rint (multi_array<T> res, multi_array<T> &rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_RINT, res, rhs);
    return res;
}

// TODO: Fix this hack-slash support for BH_REAL/IMAG
template <typename InT, typename OutT>
multi_array<OutT>& real (multi_array<InT> &rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT,InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_REAL, *result, rhs);
    return *result;
}
template <typename InT, typename OutT>
multi_array<OutT>& imag (multi_array<InT> &rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT,InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IMAG, *result, rhs);
    return *result;
}

}
#endif

