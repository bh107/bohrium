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
#include <limits>

namespace bxx {

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

//
// Random number generators
//

//
// Sugar

inline void rand_set_seed(uint64_t seed)
{
    Runtime::instance().setRandSeed(seed);
}

inline uint64_t rand_get_seed(void)
{
    return Runtime::instance().getRandSeed();
}

inline void rand_set_state(uint64_t state)
{
    Runtime::instance().setRandState(state);
}

inline uint64_t rand_get_state(void)
{
    return Runtime::instance().getRandState();
}

template <typename T, typename ...Dimensions>
multi_array<T>& random(const Dimensions&... shape)
{
    int64_t nelements = nelements_shape(shape...);
    
    uint64_t state  = rand_get_state();
    uint64_t key    = rand_get_seed();

    rand_set_state(state+nelements);
                                                            // Generate numbers
    multi_array<uint64_t>* rand_result = new multi_array<uint64_t>(nelements);
    rand_result->link();
    bh_random(*rand_result, state, key);
    rand_result->setTemp(true);
    
    multi_array<T>* result = new multi_array<T>(nelements); // Convert their type
    result->link();
    bh_identity(*result, *rand_result);
    result->setTemp(true);

    return view_as(*result, shape...);                      // Reshape them
}

template <typename T, typename ...Dimensions>
multi_array<T>& randu(const Dimensions&... shape)
{
    int64_t nelements = nelements_shape(shape...);
    
    uint64_t state  = rand_get_state();
    uint64_t key    = rand_get_seed();

    rand_set_state(state+nelements);
                                                            // Generate numbers
    multi_array<uint64_t>* rand_result = new multi_array<uint64_t>(nelements);
    rand_result->link();
    bh_random(*rand_result, state, key);
    rand_result->setTemp(true);
    
    multi_array<T>* result = new multi_array<T>(nelements); // Convert their type
    result->link();
    bh_identity(*result, *rand_result);

    bh_divide(
        *result,
        *result,
        (T)std::numeric_limits<uint64_t>::max());           // Map to [0,1]

    result->setTemp(true);
    return view_as(*result, shape...);                      // Reshape them
}

//
// End of random number generators.
//

/**
 *  Create a range of values defined as the [start, end[
 *  Each element in the range is seperated by 'skip'.
 */
template <typename T>
multi_array<T>& range(const int64_t start, const int64_t end, const int64_t skip)
{
    int64_t adj_end = end - 1;
    if ((start > end) && (skip>0)) {
        throw std::runtime_error("Error: Invalid range [start>end when skip>0].");
    } else if((start < adj_end) && (skip<0)) {
        throw std::runtime_error("Error: Invalid range [start<end when skip<0].");
    } else if (skip==0) {
        throw std::runtime_error("Error: Invalid range [skip=0].");
    } else if (start==adj_end) {
        throw std::runtime_error("Error: Invalid range [start=end].");
    }

    uint64_t nelem;
    if (skip>0) {
        nelem = (adj_end-start+1)/skip;
    } else {
        nelem = (start-adj_end+1)/abs(skip);
    }

    multi_array<T>* result = new multi_array<T>(nelem);     // Construct the result
    result->link();

    if (nelem >= std::numeric_limits<uint32_t>::max()) {    // Construct the range using uint64
        multi_array<uint64_t>* base_range = new multi_array<uint64_t>(nelem);
        base_range->link();
        bh_range(*base_range);
        base_range->setTemp(true);
        bh_identity(*result, *base_range);                  // Convert the type to T
    } else {                                                // Construct the range using uint32
        multi_array<uint32_t>* base_range = new multi_array<uint32_t>(nelem);
        base_range->link();
        bh_range(*base_range);
        base_range->setTemp(true);
        bh_identity(*result, *base_range);                  // Convert the type to T
    }

    bh_multiply(*result, *result, (T)skip);                 // Expand the range
    bh_add(*result, *result, (T)start);
   
    result->setTemp(true);
    return *result;
}

/**
 *  Create a range of values defined as [0, nelem[
 */
template <typename T>
multi_array<T>& range(uint64_t nelem)
{
    return range<T>((int64_t)0, (int64_t)nelem, (int64_t)1);
}

template <typename T>
multi_array<T>& linspace(int64_t begin, int64_t end, uint64_t nelem, bool endpoint)
{
    T dist = std::abs(begin - end);
    T skip = endpoint ? dist/(T)(nelem-1) : dist/(T)nelem;

    multi_array<T>* result = new multi_array<T>(nelem);     // Construct the result
    result->link();

    if (nelem >= std::numeric_limits<uint32_t>::max()) {    // Construct the range using uint64
        multi_array<uint64_t>* base_range = new multi_array<uint64_t>(nelem);
        base_range->link();
        bh_range(*base_range);
        base_range->setTemp(true);
        bh_identity(*result, *base_range);                  // Convert the type to T
    } else {                                                // Construct the range using uint32
        multi_array<uint32_t>* base_range = new multi_array<uint32_t>(nelem);
        base_range->link();
        bh_range(*base_range);
        base_range->setTemp(true);
        bh_identity(*result, *base_range);                  // Convert the type to T
    }

    // Expand the range
    bh_multiply(*result, *result, skip);
    bh_add(*result, *result, (T)begin);
    
    result->setTemp(true);
    return *result;
}

}
#endif

