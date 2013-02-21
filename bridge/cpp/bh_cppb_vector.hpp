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
namespace bh {

template <typename T>
Vector<T>::Vector( Vector<T> const& vector )
{
    this->array = new bh_array;

    this->array->type        = vector.array->type;
    this->array->base        = vector.array->base;
    this->array->ndim        = vector.array->ndim;
    this->array->start       = vector.array->start;
    for(bh_index i=0; i< vector.array->ndim; i++) {
        this->array->shape[i] = vector.array->shape[i];
    }
    for(bh_index i=0; i< vector.array->ndim; i++) {
        this->array->stride[i] = vector.array->stride[i];
    }
    this->array->data        = vector.array->data;
}

template <typename T>
Vector<T>::Vector( int d0 )
{
    this->array = new bh_array;

    assign_array_type<T>( this->array );
    this->array->base        = NULL;
    this->array->ndim        = 1;
    this->array->start       = 0;
    this->array->shape[0]    = d0;
    this->array->stride[0]   = 1;
    this->array->data        = NULL;
}

template <typename T>
Vector<T>::Vector( int d0, int d1 )
{
    this->array = new bh_array;

    assign_array_type<T>( this->array );
    this->array->base        = NULL;
    this->array->ndim        = 2;
    this->array->start       = 0;
    this->array->shape[0]    = d0;
    this->array->stride[0]   = d1;
    this->array->shape[1]    = d1;
    this->array->stride[1]   = 1;
    this->array->data        = NULL;
}

template <typename T>
Vector<T>::~Vector()
{
    enqueue_aa( (bh_opcode)BH_FREE, *this, *this);
    enqueue_aa( (bh_opcode)BH_DISCARD, *this, *this);

    //delete this->array;
}

template <typename T>
Vector<T>& Vector<T>::operator++()
{
    std::cout << this << ": ++ v{ " << this << " }" << std::endl;
    enqueue_aac( (bh_opcode)BH_ADD, *this, *this, (T)1 );
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator++(int)
{
    std::cout << this << ": v{ " << this << " } ++" << std::endl;
    enqueue_aac( (bh_opcode)BH_ADD, *this, *this, (T)1 );
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator--()
{
    std::cout << this << ": -- v{ " << this << " }" << std::endl;
    enqueue_aac( (bh_opcode)BH_SUBTRACT, *this, *this, (T)1 );
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator--(int)
{
    std::cout << this << ": v{ " << this << " } --" << std::endl;
    enqueue_aac( (bh_opcode)BH_SUBTRACT, *this, *this, (T)1 );
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator+=( const T rhs )
{
    std::cout << this << ": += v{ " << this << " } " << std::endl;
    enqueue_aac( (bh_opcode)BH_ADD, *this, *this, rhs );
    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator+=( Vector & rhs )
{
    std::cout << this << ": += v{ " << this << " } " << std::endl;
    enqueue_aaa( (bh_opcode)BH_ADD, *this, *this, rhs );
    return *this;
}

}

