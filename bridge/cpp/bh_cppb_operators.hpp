/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef __BOHRIUM_BRIDGE_CPP_OPERATORS
#define __BOHRIUM_BRIDGE_CPP_OPERATORS
#include "cphvb.h"

namespace bh {



template <typename T>
Vector<T> & operator + ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " + " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_ADD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator + ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " + " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_ADD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator + ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " + " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_ADD, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator - ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " - " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator - ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " - " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator - ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " - " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator * ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " * " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator * ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " * " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator * ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " * " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator / ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " / " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_DIVIDE, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator / ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " / " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_DIVIDE, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator / ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " / " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_DIVIDE, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator % ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " % " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_MOD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator % ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " % " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_MOD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator % ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " % " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_MOD, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator == ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " == " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator == ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " == " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator == ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " == " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator != ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " != " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator != ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " != " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator != ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " != " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator > ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " > " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_GREATER, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator > ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " > " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_GREATER, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator > ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " > " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_GREATER, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator >= ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " >= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >= ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " >= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >= ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " >= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator < ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " < " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_LESS, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator < ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " < " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_LESS, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator < ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " < " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_LESS, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator <= ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " <= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator <= ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " <= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator <= ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " <= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator && ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " && " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator && ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " && " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator && ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " && " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator || ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " || " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator || ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " || " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator || ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " || " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator & ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " & " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator & ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " & " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator & ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " & " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator | ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " | " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator | ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " | " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator | ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " | " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator ^ ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " ^ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ^ ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " ^ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ^ ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " ^ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator << ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " << " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator << ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " << " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator << ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " << " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator >> ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " >> " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (cphvb_opcode)CPHVB_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >> ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " >> " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (cphvb_opcode)CPHVB_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >> ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " >> " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (cphvb_opcode)CPHVB_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ! ( Vector<T> & vector )
{
    std::cout << vector << " ! " << std::endl;
    enqueue_ac( (cphvb_opcode)CPHVB_LOGICAL_NOT, vector, vector );
    return vector;
}

template <typename T>
Vector<T> & operator ~ ( Vector<T> & vector )
{
    std::cout << vector << " ~ " << std::endl;
    enqueue_ac( (cphvb_opcode)CPHVB_INVERT, vector, vector );
    return vector;
}


}

#endif
