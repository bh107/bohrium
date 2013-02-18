#ifndef __BOHRIUM_BRIDGE_CPP_OPERATORS
#define __BOHRIUM_BRIDGE_CPP_OPERATORS
#include "bh.h"

namespace bh {



template <typename T>
Vector<T> & operator + ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " + " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_ADD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator + ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " + " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_ADD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator + ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " + " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_ADD, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator - ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " - " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator - ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " - " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator - ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " - " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_SUBTRACT, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator * ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " * " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator * ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " * " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator * ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " * " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_MULTIPLY, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator / ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " / " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_DIVIDE, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator / ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " / " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_DIVIDE, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator / ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " / " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_DIVIDE, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator % ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " % " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_MOD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator % ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " % " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_MOD, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator % ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " % " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_MOD, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator == ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " == " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator == ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " == " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator == ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " == " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator != ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " != " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator != ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " != " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator != ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " != " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_NOT_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator > ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " > " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_GREATER, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator > ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " > " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_GREATER, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator > ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " > " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_GREATER, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator >= ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " >= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >= ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " >= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >= ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " >= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_GREATER_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator < ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " < " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LESS, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator < ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " < " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LESS, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator < ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " < " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LESS, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator <= ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " <= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator <= ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " <= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator <= ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " <= " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LESS_EQUAL, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator && ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " && " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator && ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " && " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator && ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " && " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LOGICAL_AND, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator || ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " || " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator || ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " || " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator || ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " || " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LOGICAL_OR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator & ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " & " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator & ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " & " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator & ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " & " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_BITWISE_AND, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator | ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " | " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator | ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " | " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator | ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " | " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_BITWISE_OR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator ^ ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " ^ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ^ ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " ^ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ^ ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " ^ " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_BITWISE_XOR, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator << ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " << " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator << ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " << " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator << ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " << " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_LEFT_SHIFT, *vector, lhs, rhs );
    return *vector;
}


template <typename T>
Vector<T> & operator >> ( Vector<T> & lhs, Vector<T> & rhs )
{
    std::cout << lhs << " >> " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aaa( (bh_opcode)BH_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >> ( Vector<T> & lhs, T const& rhs )
{
    std::cout << lhs << " >> " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( lhs );
    enqueue_aac( (bh_opcode)BH_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator >> ( T const& lhs, Vector<T> & rhs )
{
    std::cout << lhs << " >> " << rhs << std::endl;
    Vector<T>* vector = new Vector<T>( rhs );
    enqueue_aca( (bh_opcode)BH_RIGHT_SHIFT, *vector, lhs, rhs );
    return *vector;
}

template <typename T>
Vector<T> & operator ! ( Vector<T> & vector )
{
    std::cout << vector << " ! " << std::endl;
    enqueue_ac( (bh_opcode)BH_LOGICAL_NOT, vector, vector );
    return vector;
}

template <typename T>
Vector<T> & operator ~ ( Vector<T> & vector )
{
    std::cout << vector << " ~ " << std::endl;
    enqueue_ac( (bh_opcode)BH_INVERT, vector, vector );
    return vector;
}


}

#endif
