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
#ifndef __BOHRIUM_BRIDGE_CPP_GENERATOR
#define __BOHRIUM_BRIDGE_CPP_GENERATOR

namespace bh {

template <typename T, typename ...Dimensions>
multi_array<T>& value(T val, const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->setTemp(true);
    result->link();
    *result = val;

    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& empty(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->setTemp(true);
    result->link();

    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& ones(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->setTemp(true);

    result->link();
    *result = (T)1;

    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& zeros(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->setTemp(true);

    result->link();
    *result = (T)0;

    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& random(const Dimensions&... shape)
{
    bh_random_type* rinstr = (bh_random_type*)malloc(sizeof(bh_random_type));
    if (rinstr == NULL) {
        char err_msg[100];
        sprintf(err_msg, "Failed alllocating memory for extension-call.");
        throw std::runtime_error(err_msg);
    }

    multi_array<T>* result = new multi_array<T>(shape...);
    result->setTemp(true);
    result->link();

    rinstr->id          = Runtime::instance().random_id;        //Set the instruction
    rinstr->nout        = 1;
    rinstr->nin         = 0;
    rinstr->struct_size = sizeof(bh_random_type);
    rinstr->operand[0]  = result->meta;

    Runtime::instance().enqueue<T>((bh_userfunc*)rinstr);

    return *result;
}

template <typename T>
multi_array<T>& random(const int64_t rank, const int64_t* shape)
{
    bh_random_type* rinstr = (bh_random_type*)malloc(sizeof(bh_random_type));
    if (rinstr == NULL) {
        char err_msg[100];
        sprintf(err_msg, "Failed alllocating memory for extension-call.");
        throw std::runtime_error(err_msg);
    }

    multi_array<T>* result = new multi_array<T>(rank, shape);
    result->setTemp(true);
    result->link();

    rinstr->id          = Runtime::instance().random_id;        //Set the instruction
    rinstr->nout        = 1;
    rinstr->nin         = 0;
    rinstr->struct_size = sizeof(bh_random_type);
    rinstr->operand[0]  = result->meta;

    Runtime::instance().enqueue<T>((bh_userfunc*)rinstr);

    return *result;
}

template <typename T>
multi_array<T>& arange(const int64_t start, const int64_t end, const int64_t skip)
{
    const int64_t tmp[] = { end - start };
    multi_array<T>* result = new multi_array<T>((const int64_t)1, tmp);
    result->setTemp(true);
    result->link();

    for(int64_t i = 0; i < (end-start); i++) {
        (*result)[i] = (T)i;
    }

    *result *= skip;
    *result += start;

    std::cout << "arange(" << start << "," << end << "," << skip << ");" << std::endl;

    return *result;
}

}
#endif

