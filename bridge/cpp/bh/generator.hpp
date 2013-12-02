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
    result->link();

    *result = val;

    result->setTemp(true);
    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& empty(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->link();

    result->setTemp(true);
    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& ones(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->link();

    *result = (T)1;

    result->setTemp(true);
    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& zeros(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->link();

    *result = (T)0;

    result->setTemp(true);
    return *result;
}

#ifndef NO_VARIADICS
template <typename T, typename ...Dimensions>
multi_array<T>& random(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RANGE,    *result);
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *result, (T)1);
    Runtime::instance().enqueue((bh_opcode)BH_ADD,      *result, *result, (T)0);
    Runtime::instance().enqueue((bh_opcode)BH_RANDOM,   *result, (T)time(NULL), *result);
    
    result->setTemp(true);
    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& randu(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RANGE,    *result);
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *result, (T)1);
    Runtime::instance().enqueue((bh_opcode)BH_ADD,      *result, *result, (T)0);
    Runtime::instance().enqueue((bh_opcode)BH_RANDOM,   *result, (T)time(NULL), *result);
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE,   *result, (T)1, *result);
    
    result->setTemp(true);
    return *result;
}

template <typename T, typename ...Dimensions>
multi_array<T>& randn(const Dimensions&... shape)
{
    multi_array<T>* result = new multi_array<T>(shape...);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RANGE,    *result);
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *result, (T)1);
    Runtime::instance().enqueue((bh_opcode)BH_ADD,      *result, *result, (T)0);
    Runtime::instance().enqueue((bh_opcode)BH_RANDOM,   *result, (T)time(NULL), *result);
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE,   *result, (T)1, *result);
    
    result->setTemp(true);
    return *result;
}
#endif

template <typename T>
multi_array<T>& random(const int64_t length)
{
    multi_array<T>* result = new multi_array<T>(1, &length);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RANGE,    *result);
    Runtime::instance().enqueue((bh_opcode)BH_ADD,      *result, *result, (T)0);
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *result, (T)1);
    Runtime::instance().enqueue((bh_opcode)BH_RANDOM,   *result, (T)42, *result);
    
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<T>& range(const int64_t start, const int64_t end, const int64_t skip)
{
    if ((start > end) && (skip>0)) {
        throw std::runtime_error("Error: Invalid range [start>end when skip>0].");
    } else if((start < end) && (skip<0)) {
        throw std::runtime_error("Error: Invalid range [start<end when skip<0].");
    } else if (skip==0) {
        throw std::runtime_error("Error: Invalid range [skip=0].");
    } else if (start==end) {
        throw std::runtime_error("Error: Invalid range [start=end].");
    }
    
    int64_t nelem;
    if (skip>0) {
        nelem = (end-start+1)/skip;
    } else {
        nelem = (start-end+1)/abs(skip);
    }

    multi_array<T>* result = new multi_array<T>(1, &nelem);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RANGE,    *result);
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *result, (T)skip);
    Runtime::instance().enqueue((bh_opcode)BH_ADD,      *result, *result, (T)start);

    result->setTemp(true);
    return *result;
}

}
#endif

