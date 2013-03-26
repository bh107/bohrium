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
#ifndef __BOHRIUM_BRIDGE_CPP_EXTENSIONS
#define __BOHRIUM_BRIDGE_CPP_EXTENSIONS

namespace bh {

template <typename T>
multi_array<T>& multi_array<T>::reduce(reducible op, int axis)
{
    char err_msg[100];

    bh_reduce_type* rinstr;
    multi_array<T>* result = &Runtime::instance()->temp(*this);
    
    rinstr = (bh_reduce_type*)malloc(sizeof(bh_reduce_type)); //Allocate the user-defined function.
    if (rinstr == NULL) {
        sprintf(err_msg, "Failed alllocating memory for extension-call.");
        throw std::runtime_error(err_msg);
    }
    
    rinstr->id          = reduce_id;        //Set the instruction
    rinstr->nout        = 1;
    rinstr->nin         = 1;
    rinstr->struct_size = sizeof(bh_reduce_type);
    rinstr->operand[0]  = &storage[result->getKey()];
    rinstr->operand[1]  = &storage[key];
    rinstr->opcode      = (bh_opcode)op;
    rinstr->axis        = axis;

    Runtime::instance()->enqueue<T>((bh_userfunc*)rinstr);

    return *result;
}

template <typename T>
multi_array<T>& random(int n)
{
    multi_array<T>* result = new multi_array<T>(n);

    char err_msg[100];

    bh_random_type* rinstr;
    
    rinstr = (bh_random_type*)malloc(sizeof(bh_random_type)); //Allocate the user-defined function.
    if (rinstr == NULL) {
        sprintf(err_msg, "Failed alllocating memory for extension-call.");
        throw std::runtime_error(err_msg);
    }
    
    rinstr->id          = random_id;        //Set the instruction
    rinstr->nout        = 1;
    rinstr->nin         = 0;
    rinstr->struct_size = sizeof(bh_random_type);
    rinstr->operand[0]  = &storage[result->getKey()];

    Runtime::instance()->enqueue<T>((bh_userfunc*)rinstr);

    return *result;
}

}
#endif
