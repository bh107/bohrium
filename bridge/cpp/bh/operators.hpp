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
multi_array<T>& multi_array<T>::operator= ( T const& rhs )
{
    Runtime::instance()->enqueue( (bh_opcode)BH_IDENTITY, *this, rhs );
    return *this;
}


template <typename T>
multi_array<T>& multi_array<T>::operator = ( multi_array<T> & rhs )
{
    Runtime::instance()->enqueue( (bh_opcode)BH_IDENTITY, *this, rhs );
    return *this;
}

//
//  Binary and implemented by code-generator.
//  Operators such as:
//  Mapping "a + b" to BH_ADD(t, a, b)
//

template <typename T>
multi_array<T> & operator + ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_ADD, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator + ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_ADD, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator + ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_ADD, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator - ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_SUBTRACT, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator - ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_SUBTRACT, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator - ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_SUBTRACT, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator * ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_MULTIPLY, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator * ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_MULTIPLY, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator * ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_MULTIPLY, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator / ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_DIVIDE, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator / ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_DIVIDE, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator / ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_DIVIDE, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator % ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_MOD, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator % ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_MOD, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator % ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_MOD, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator == ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator == ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator == ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator != ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_NOT_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator != ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_NOT_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator != ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_NOT_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator > ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_GREATER, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator > ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_GREATER, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator > ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_GREATER, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator >= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_GREATER_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator >= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_GREATER_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator >= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_GREATER_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator < ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_LESS, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator < ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LESS, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator < ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LESS, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator <= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_LESS_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator <= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LESS_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator <= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LESS_EQUAL, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator && ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_LOGICAL_AND, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator && ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LOGICAL_AND, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator && ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LOGICAL_AND, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator || ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_LOGICAL_OR, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator || ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LOGICAL_OR, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator || ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LOGICAL_OR, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator & ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_AND, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator & ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_AND, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator & ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_AND, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator | ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_OR, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator | ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_OR, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator | ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_OR, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator ^ ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);

    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_XOR, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator ^ ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_XOR, *operand, lhs, rhs );

    return *operand;
}

template <typename T>
multi_array<T> & operator ^ ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_BITWISE_XOR, *operand, lhs, rhs );

    return *operand;
}

/*
//
//  Binary and implemented by manually.
//  Operators such as:
//  None so far...
//

template <typename T>
multi_array<T> & operator += ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator += ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator += ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator -= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator -= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator -= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator *= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator *= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator *= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator /= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator /= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator /= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator %= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator %= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator %= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator &= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator &= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator &= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator |= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator |= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator |= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator ^= ( multi_array<T> & lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator ^= ( multi_array<T> & lhs, T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( lhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}

template <typename T>
multi_array<T> & operator ^= ( T const& lhs, multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    // TODO: implement
    return *operand;
}
*/

//
//  Unary and implemented by code-generator.
//  Operators such as:
//  Mapping "!a" to BH_NEGATE(t, a)
//

template <typename T>
multi_array<T> & operator ! ( multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LOGICAL_NOT, *operand, rhs );

    return *operand;
}

/*
template <typename T>
multi_array<T> & operator ! ( T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_LOGICAL_NOT, *operand, rhs );

    return *operand;
}
*/


template <typename T>
multi_array<T> & operator ~ ( multi_array<T> & rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_INVERT, *operand, rhs );

    return *operand;
}

/*
template <typename T>
multi_array<T> & operator ~ ( T const& rhs )
{
    multi_array<T>* operand = new multi_array<T>( rhs );
    operand->setTemp(true);
    Runtime::instance()->enqueue( (bh_opcode)BH_INVERT, *operand, rhs );

    return *operand;
}
*/


/*
//
//  Unary and implemented manually.
//  Operators such as:
//  Mapping "++a" to BH_ADD(a, a, 1)
//

*/
}
#endif
