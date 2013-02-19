#ifndef __BOHRIUM_BRIDGE_CPP_OPERATORS
#define __BOHRIUM_BRIDGE_CPP_OPERATORS
#include "bh.h"

namespace bh {


//
//  Binary
//  Externally defined
//  Directly mapped opcode to c++ operators such as:
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
//  Binary
//  Externally defined
//  Custom-mapping opcode to c++ operators such as:
//  Hmmm dunnoo...
//

//
//  Unary
//  Externally defined
//  Directly mapping opcode to c++ operators such as:
//  sin( a ) to BH_SIN(t, a)
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
//  Unary
//  Externally defined
//  Indirectly mapping opcode to c++ operators such as:
//  Mapping the "a++" operator to BH_ADD( a, a, 1)
//


}

#endif
