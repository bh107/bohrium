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

#ifndef __BH_H
#define __BH_H

#include <bh_type.h>
#include <bh_opcode.h>
#include <bh_win.h>
#include <bh_error.h>

#include <cstddef>
#include <iostream>
#include <vector>


#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) >= (b) ? (a) : (b))

/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
DLLEXPORT const char* bh_opcode_text(bh_opcode opcode);

/* Determines if the operation is a system operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_system(bh_opcode opcode);

/* Determines if the operation is an elementwise operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_elementwise(bh_opcode opcode);

/* Determines if the operation is a reduction operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_reduction(bh_opcode opcode);

/* Determines if the operation is an accumulate operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_accumulate(bh_opcode opcode);

/* Determines if the operation is performed elementwise
 *
 * @opcode Opcode for operation
 * @return TRUE if the operation is performed elementwise, FALSE otherwise
 */
DLLEXPORT bool bh_opcode_is_elementwise(bh_opcode opcode);

/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
DLLEXPORT int bh_type_size(bh_type type);

/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
DLLEXPORT const char* bh_type_text(bh_type type);

/* Is type an integer type
 *
 * @type   The type.
 * @return True if integer type.
 */
DLLEXPORT bool bh_type_is_integer(bh_type type);

/* Determines whether the opcode is a sweep opcode
 * i.e. either a reduction or an accumulate
 *
 * @opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_sweep(bh_opcode opcode);

template<typename E>
std::ostream& operator<<(std::ostream& out, const std::vector<E>& v)
{
    out << "[";
    for (typename std::vector<E>::const_iterator i = v.cbegin();;)
    {
        out << *i;
        if (++i == v.cend())
            break;
        out << ", ";
    }
    out << "]";
    return out;
}

#endif
