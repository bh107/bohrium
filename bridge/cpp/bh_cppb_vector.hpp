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
namespace cphvb {

template <typename T>
Vector<T>::Vector( Vector<T> const& vector )
{
    this->array = new cphvb_array;

    this->array->type        = vector.array->type;
    this->array->base        = vector.array->base;
    this->array->ndim        = vector.array->ndim;
    this->array->start       = vector.array->start;
    for(cphvb_index i=0; i< vector.array->ndim; i++) {
        this->array->shape[i] = vector.array->shape[i];
    }
    for(cphvb_index i=0; i< vector.array->ndim; i++) {
        this->array->stride[i] = vector.array->stride[i];
    }
    this->array->data        = vector.array->data;
}

template <typename T>
Vector<T>::Vector( int d0 )
{
    this->array = new cphvb_array;

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
    this->array = new cphvb_array;

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

}

