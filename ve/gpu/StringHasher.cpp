/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

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
 
#include "StringHasher.hpp"

#ifdef _WIN64
#define HAS_CPP_HASH 0
#elif _WIN32
#define HAS_CPP_HASH 0
#elif __APPLE__
#define HAS_CPP_HASH 0
#elif __cplusplus >= 199711L
#define HAS_CPP_HASH 1
#else
#define HAS_CPP_HASH 0
#endif

#if HAS_CPP_HASH > 0
#include <functional>
std::hash<std::string> hash_fn; 
size_t string_hasher(std::string str)
{
    return hash_fn(str);
}
#else
size_t string_hasher(std::string str)
{
    //fnv_64_str fixed to use fnv-1 initial value
    bh_int64 hval = 0xcbf29ce484222325ULL;
    const unsigned char *s = (const unsigned char *)str.c_str();
    while (*s) {
        /* multiply by the 64 bit FNV magic prime mod 2^64 */
        hval += (hval << 1) + (hval << 4) + (hval << 5) +
            (hval << 7) + (hval << 8) + (hval << 40);
        /* xor the bottom with the current octet */
        hval ^= (bh_int64)*s++;
    }
    return (size_t)hval;
}
#endif

#undef HAS_CPP_HASH
