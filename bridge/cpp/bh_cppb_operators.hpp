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
#ifndef __BOHRIUM_BRIDGE_CPP_OPERATORS
#define __BOHRIUM_BRIDGE_CPP_OPERATORS
#include "bh.h"

namespace bh {


//
//  Internally defined operator overloads
//


template <typename T>
Vector<T>& Vector<T>::operator = ( const T rhs )
{
    //std::cout << *this << " = " << rhs << std::endl;
    enqueue_ac( (bh_opcode)BH_IDENTITY, *this, rhs );
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator = ( Vector<T> & rhs )
{
    //std::cout << *this << " = " << rhs << std::endl;
    enqueue_aa( (bh_opcode)BH_IDENTITY, *this, rhs );
    return *this;
}



template <typename T>
Vector<T>& Vector<T>::operator [] ( const T rhs )
{
    //std::cout << *this << " [] " << rhs << std::endl;
    // TODO: implement
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator [] ( Vector<T> & rhs )
{
    //std::cout << *this << " [] " << rhs << std::endl;
    // TODO: implement
    return *this;
}


template <typename T>
Vector<T>& Vector<T>::operator () ( const T rhs )
{
    //std::cout << *this << " () " << rhs << std::endl;
    // TODO: implement
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator () ( Vector<T> & rhs )
{
    //std::cout << *this << " () " << rhs << std::endl;
    // TODO: implement
    return *this;
}


template <typename T>
Vector<T>& Vector<T>::operator -> ( const T rhs )
{
    //std::cout << *this << " -> " << rhs << std::endl;
    // TODO: implement
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator -> ( Vector<T> & rhs )
{
    //std::cout << *this << " -> " << rhs << std::endl;
    // TODO: implement
    return *this;
}



//
//  Binary and implemented by code-generator.
//  Operators such as:
//  Mapping "a + b" to BH_ADD(t, a, b)
//

template <typename T>
Vector<T> & operator + ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " + " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_ADD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator + ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_ADD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator + ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_ADD, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator - ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " - " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator - ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator - ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator * ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " * " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator * ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator * ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator / ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " / " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_DIVIDE, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator / ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_DIVIDE, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator / ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_DIVIDE, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator % ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " % " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_MOD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator % ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_MOD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator % ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_MOD, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator == ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " == " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator == ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator == ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator != ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " != " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator != ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator != ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator > ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " > " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_GREATER, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator > ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_GREATER, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator > ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_GREATER, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator >= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " >= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >= ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >= ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator < ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " < " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LESS, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator < ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LESS, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator < ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LESS, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator <= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " <= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator <= ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator <= ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator && ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " && " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator && ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator && ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator || ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " || " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator || ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator || ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator & ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " & " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator & ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator & ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator | ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " | " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator | ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator | ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator ^ ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " ^ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ^ ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ^ ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator << ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " << " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator << ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator << ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator >> ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << lhs << " >> " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >> ( Vector<T> & lhs, T const& rhs )
{
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >> ( T const& lhs, Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}


//
//  Binary and implemented by manually.
//  Operators such as:
//  None so far...
//

template <typename T>
Vector<T> & operator += ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " += " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator += ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " += " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator += ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " += " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator -= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " -= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator -= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " -= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator -= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " -= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator *= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " *= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator *= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " *= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator *= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " *= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator /= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " /= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator /= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " /= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator /= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " /= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator %= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " %= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator %= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " %= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator %= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " %= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator &= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " &= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator &= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " &= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator &= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " &= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator |= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " |= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator |= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " |= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator |= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " |= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator ^= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " ^= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator ^= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " ^= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator ^= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " ^= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator <<= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " <<= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator <<= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " <<= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator <<= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " <<= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator >>= ( Vector<T> & lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " >>= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator >>= ( Vector<T> & lhs, T const& rhs )
{
    //std::cout << "<UNKNOWN>" << " >>= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator >>= ( T const& lhs, Vector<T> & rhs )
{
    //std::cout << "<UNKNOWN>" << " >>= " << "<UNKNOWN>" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


//
//  Binary and implemented by code-generator.
//  Operators such as:
//  Mapping "!a" to BH_ADD(t, a, b)
//

template <typename T>
Vector<T> & operator ! ( Vector<T> & rhs )
{
    //std::cout << " ! " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aa( (bh_opcode)BH_LOGICAL_NOT, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ! ( T const& rhs )
{
    //std::cout << " ! " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_ac( (bh_opcode)BH_LOGICAL_NOT, *vector, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator ~ ( Vector<T> & rhs )
{
    //std::cout << " ~ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aa( (bh_opcode)BH_INVERT, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ~ ( T const& rhs )
{
    //std::cout << " ~ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_ac( (bh_opcode)BH_INVERT, *vector, rhs );
    return *vector;
}


//
//  Unary and implemented manually.
//  Operators such as:
//  Mapping "++a" to BH_ADD(a, a, 1)
//

template <typename T>
Vector<T> & operator ++ ( Vector<T> & rhs )
{
    //std::cout << " ++ " << "UNK" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator ++ ( T const& rhs )
{
    //std::cout << " ++ " << "UNK" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator ++ ( Vector<T> & rhs )
{
    //std::cout << " ++ " << "UNK" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator ++ ( T const& rhs )
{
    //std::cout << " ++ " << "UNK" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator -- ( Vector<T> & rhs )
{
    //std::cout << " -- " << "UNK" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator -- ( T const& rhs )
{
    //std::cout << " -- " << "UNK" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}


template <typename T>
Vector<T> & operator -- ( Vector<T> & rhs )
{
    //std::cout << " -- " << "UNK" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}

template <typename T>
Vector<T> & operator -- ( T const& rhs )
{
    //std::cout << " -- " << "UNK" << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    // TODO: implement
    return *vector;
}



}

#endif
