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

#ifndef __BH_JITK_KERNEL_DEPENDENCIES_RANDOM123_H
#define __BH_JITK_KERNEL_DEPENDENCIES_RANDOM123_H

#include <Random123/philox.h>

// This macro defines the random123 function using 'type' as input and output.
// Should be called once in JIT-kernels that uses random123()
#define DEF_RANDOM123(type)                                  \
type random123(type start, type key, type index) {           \
    union {philox2x32_ctr_t c; type ul;} ctr, res;           \
    ctr.ul = start + index;                                  \
    res.c = philox2x32(ctr.c, (philox2x32_key_t){{key}});    \
    return res.ul;                                           \
}

#endif
