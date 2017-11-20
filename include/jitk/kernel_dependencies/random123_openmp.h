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

// This is the C99/OpenMP interface to Random123
#pragma once

#include <Random123/philox.h>

philox2x32_ctr_t philox2x32_R(unsigned int R, philox2x32_ctr_t ctr, philox2x32_key_t key);

uint64_t random123(uint64_t start, uint64_t key, uint64_t index) {
    uint64_t i = start + index;

    philox2x32_ctr_t _index = *((philox2x32_ctr_t*)&i);
    philox2x32_key_t _key   = *((philox2x32_key_t*)&key);

    philox2x32_ctr_t result = philox2x32_R(philox2x32_rounds, _index, _key);

    return *((uint64_t*)&result);
}
