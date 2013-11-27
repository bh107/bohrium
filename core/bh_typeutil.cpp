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
#include <bh.h>

/* Determines if the types are acceptable for the operation
 *
 * @opcode Opcode for operation
 * @outtype The type of the output
 * @inputtype1 The type of the first input operand
 * @inputtype2 The type of the second input operand
 * @constanttype The type of the constant
 * @return TRUE if the types are accepted, FALSE otherwise
 */
bool bh_validate_types(bh_opcode opcode, bh_type outtype, bh_type inputtype1, bh_type inputtype2, bh_type constanttype)
{
    // Poly contains a unique value, pairing an opcode with its function signature.
    // All in one nicely switchable value.
    long int poly;
 
    if (bh_operands(opcode) == 3) {                 // Three operands

        if (inputtype1 == BH_UNKNOWN) {             // First operand is constant
            poly = opcode \
                | (outtype << 8) \
                | (constanttype << 12) \
                | (inputtype2 << 16);

        } else if (inputtype2 == BH_UNKNOWN) {      // Second operand is constant
            poly = opcode
                | (outtype << 8) \
                | (inputtype1 << 12) \
                | (constanttype << 16);

        } else {                                     // No constant operand
            poly = opcode \
                | (outtype << 8) \
                | (inputtype1 << 12) \
                | (inputtype2 << 16);
        }

    } else {                                         // Two operands

        if (inputtype1 == BH_UNKNOWN) {
            poly = opcode \
                | (outtype << 8) \
                | (constanttype << 12) \
                | (1 << 17);
        } else {
            poly = opcode \
                | (outtype << 8) \
                | (inputtype1 << 12) \
                | (1 << 17);
        }
    }
    

    switch (poly)
    {
        case BH_ADD | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_ADD | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_ADD | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_ADD | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_ADD | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_ADD | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_ADD | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_ADD | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_ADD | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_ADD | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_ADD | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_ADD | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_ADD | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_SUBTRACT | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_SUBTRACT | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_SUBTRACT | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_SUBTRACT | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_SUBTRACT | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_SUBTRACT | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_SUBTRACT | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_SUBTRACT | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_SUBTRACT | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_SUBTRACT | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_SUBTRACT | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_SUBTRACT | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_SUBTRACT | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_MULTIPLY | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_MULTIPLY | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_MULTIPLY | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_MULTIPLY | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_MULTIPLY | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_MULTIPLY | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_MULTIPLY | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_MULTIPLY | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_MULTIPLY | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_MULTIPLY | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_MULTIPLY | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_MULTIPLY | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_MULTIPLY | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_DIVIDE | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_DIVIDE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_DIVIDE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_DIVIDE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_DIVIDE | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_DIVIDE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_DIVIDE | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_DIVIDE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_DIVIDE | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_DIVIDE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_DIVIDE | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_DIVIDE | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_POWER | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_POWER | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_POWER | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_POWER | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_POWER | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_POWER | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_POWER | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_POWER | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_POWER | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_POWER | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_ABSOLUTE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_GREATER | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_LOGICAL_AND | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LOGICAL_OR | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LOGICAL_XOR | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LOGICAL_NOT | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_MAXIMUM | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_MAXIMUM | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_MAXIMUM | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_MAXIMUM | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_MAXIMUM | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_MAXIMUM | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_MAXIMUM | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_MAXIMUM | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_MAXIMUM | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_MAXIMUM | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_MAXIMUM | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_MINIMUM | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_MINIMUM | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_MINIMUM | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_MINIMUM | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_MINIMUM | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_MINIMUM | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_MINIMUM | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_MINIMUM | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_MINIMUM | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_MINIMUM | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_MINIMUM | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_BITWISE_AND | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_BITWISE_AND | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_BITWISE_AND | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_BITWISE_AND | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_BITWISE_AND | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_BITWISE_AND | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_BITWISE_AND | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_BITWISE_AND | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_BITWISE_AND | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_BITWISE_OR | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_BITWISE_OR | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_BITWISE_OR | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_BITWISE_OR | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_BITWISE_OR | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_BITWISE_OR | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_BITWISE_OR | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_BITWISE_OR | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_BITWISE_OR | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_BITWISE_XOR | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_BITWISE_XOR | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_BITWISE_XOR | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_BITWISE_XOR | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_BITWISE_XOR | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_BITWISE_XOR | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_BITWISE_XOR | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_BITWISE_XOR | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_BITWISE_XOR | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_INVERT | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_INVERT | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_INVERT | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_INVERT | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_INVERT | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_INVERT | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_INVERT | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_INVERT | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_INVERT | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_LEFT_SHIFT | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_LEFT_SHIFT | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_LEFT_SHIFT | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_LEFT_SHIFT | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_LEFT_SHIFT | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_LEFT_SHIFT | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_LEFT_SHIFT | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_LEFT_SHIFT | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_RIGHT_SHIFT | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_RIGHT_SHIFT | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_RIGHT_SHIFT | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_RIGHT_SHIFT | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_RIGHT_SHIFT | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_RIGHT_SHIFT | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_RIGHT_SHIFT | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_RIGHT_SHIFT | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_COS | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_COS | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_COS | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_COS | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_SIN | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_SIN | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_SIN | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_SIN | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_TAN | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_TAN | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_TAN | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_TAN | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_COSH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_COSH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_COSH | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_COSH | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_SINH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_SINH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_SINH | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_SINH | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_TANH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_TANH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_TANH | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_TANH | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_ARCSIN | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCSIN | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCCOS | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCCOS | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCTAN | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCTAN | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCSINH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCSINH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCCOSH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCCOSH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCTANH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCTANH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCTAN2 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_ARCTAN2 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_EXP | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_EXP | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_EXP | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_EXP | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_EXP2 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_EXP2 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_EXPM1 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_EXPM1 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_LOG | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_LOG | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_LOG | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_LOG | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_LOG2 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_LOG2 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_LOG10 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_LOG10 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_LOG10 | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_LOG10 | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_LOG1P | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_LOG1P | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_SQRT | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_SQRT | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_SQRT | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_SQRT | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_CEIL | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_CEIL | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_TRUNC | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_TRUNC | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_FLOOR | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_FLOOR | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_RINT | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_RINT | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_MOD | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_MOD | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_MOD | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_MOD | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_MOD | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_MOD | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_MOD | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_MOD | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_MOD | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_MOD | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_ISNAN | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ISNAN | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ISINF | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ISINF | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_LOGICAL_AND_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_LOGICAL_OR_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_LOGICAL_XOR_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
            return true;
                    
        default:
            return false;
    }

}

/* Determines if the types are acceptable for the operation, 
 * and provides a suggestion for converting them
 *
 * @opcode Opcode for operation
 * @outtype The type of the output
 * @inputtype1 The type of the first input operand
 * @inputtype2 The type of the second input operand
 * @constanttype The type of the constant
 * @return TRUE if the types can be converted, FALSE otherwise
 */
bool bh_get_type_conversion(bh_opcode opcode, bh_type outtype, bh_type* inputtype1, bh_type* inputtype2, bh_type* constanttype) 
{
    // Basic case, "it just works"
    if (bh_validate_types(opcode, outtype, *inputtype1, *inputtype2, *constanttype))
        return true;
        
    // All valid identity types are covered
    if (opcode == BH_IDENTITY)
        return false;
    
    // Grab the input types
    bh_type desired_input_type1 = BH_UNKNOWN;
    bh_type desired_input_type2 = BH_UNKNOWN;

    // Poly contains a unique value, pairing an opcode with its function signature.
    // All in one nicely switchable value.
    long int poly;
    
    poly = opcode | (outtype << 8);

    switch(poly)
    {
            case BH_ADD | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_ADD | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_ADD | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_ADD | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_ADD | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_ADD | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_ADD | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_ADD | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_ADD | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_ADD | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_ADD | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_ADD | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                desired_input_type2 = BH_COMPLEX64;
                break;
            case BH_ADD | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                desired_input_type2 = BH_COMPLEX128;
                break;
            case BH_SUBTRACT | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_SUBTRACT | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_SUBTRACT | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_SUBTRACT | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_SUBTRACT | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_SUBTRACT | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_SUBTRACT | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_SUBTRACT | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_SUBTRACT | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_SUBTRACT | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_SUBTRACT | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_SUBTRACT | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                desired_input_type2 = BH_COMPLEX64;
                break;
            case BH_SUBTRACT | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                desired_input_type2 = BH_COMPLEX128;
                break;
            case BH_MULTIPLY | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MULTIPLY | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_MULTIPLY | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_MULTIPLY | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_MULTIPLY | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_MULTIPLY | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_MULTIPLY | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_MULTIPLY | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_MULTIPLY | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_MULTIPLY | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_MULTIPLY | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_MULTIPLY | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                desired_input_type2 = BH_COMPLEX64;
                break;
            case BH_MULTIPLY | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                desired_input_type2 = BH_COMPLEX128;
                break;
            case BH_DIVIDE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_DIVIDE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_DIVIDE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_DIVIDE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_DIVIDE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_DIVIDE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_DIVIDE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_DIVIDE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_DIVIDE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_DIVIDE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_DIVIDE | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                desired_input_type2 = BH_COMPLEX64;
                break;
            case BH_DIVIDE | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                desired_input_type2 = BH_COMPLEX128;
                break;
            case BH_POWER | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_POWER | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_POWER | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_POWER | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_POWER | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_POWER | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_POWER | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_POWER | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_POWER | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_POWER | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_ABSOLUTE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_ABSOLUTE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ABSOLUTE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_ABSOLUTE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_ABSOLUTE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_ABSOLUTE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ABSOLUTE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_ABSOLUTE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_ABSOLUTE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_ABSOLUTE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_ABSOLUTE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_GREATER | (BH_BOOL << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MAXIMUM | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MAXIMUM | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_MAXIMUM | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_MAXIMUM | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_MAXIMUM | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_MAXIMUM | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_MAXIMUM | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_MAXIMUM | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_MAXIMUM | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_MAXIMUM | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_MAXIMUM | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_MINIMUM | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MINIMUM | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_MINIMUM | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_MINIMUM | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_MINIMUM | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_MINIMUM | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_MINIMUM | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_MINIMUM | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_MINIMUM | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_MINIMUM | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_MINIMUM | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_BITWISE_AND | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_BITWISE_AND | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_BITWISE_AND | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_BITWISE_AND | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_BITWISE_AND | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_BITWISE_AND | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_BITWISE_AND | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_BITWISE_AND | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_BITWISE_AND | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_BITWISE_OR | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_BITWISE_OR | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_BITWISE_OR | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_BITWISE_OR | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_BITWISE_OR | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_BITWISE_OR | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_BITWISE_OR | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_BITWISE_OR | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_BITWISE_OR | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_BITWISE_XOR | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_BITWISE_XOR | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_BITWISE_XOR | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_BITWISE_XOR | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_BITWISE_XOR | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_BITWISE_XOR | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_BITWISE_XOR | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_BITWISE_XOR | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_BITWISE_XOR | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_INVERT | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_INVERT | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_INVERT | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_INVERT | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_INVERT | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_INVERT | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_INVERT | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_INVERT | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_INVERT | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_LEFT_SHIFT | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_LEFT_SHIFT | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_LEFT_SHIFT | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_LEFT_SHIFT | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_LEFT_SHIFT | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_LEFT_SHIFT | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_LEFT_SHIFT | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_LEFT_SHIFT | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_RIGHT_SHIFT | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_RIGHT_SHIFT | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_RIGHT_SHIFT | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_RIGHT_SHIFT | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_RIGHT_SHIFT | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_RIGHT_SHIFT | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_RIGHT_SHIFT | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_RIGHT_SHIFT | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_COS | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_COS | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_COS | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_COS | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_SIN | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_SIN | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_SIN | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_SIN | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_TAN | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_TAN | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_TAN | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_TAN | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_COSH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_COSH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_COSH | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_COSH | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_SINH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_SINH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_SINH | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_SINH | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_TANH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_TANH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_TANH | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_TANH | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_ARCSIN | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCSIN | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCCOS | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCCOS | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCTAN | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCTAN | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCSINH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCSINH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCCOSH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCCOSH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCTANH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCTANH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCTAN2 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_ARCTAN2 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_EXP | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_EXP | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_EXP | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_EXP | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_EXP2 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_EXP2 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_EXPM1 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_EXPM1 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_LOG | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_LOG | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_LOG | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_LOG | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_LOG2 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_LOG2 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_LOG10 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_LOG10 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_LOG10 | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_LOG10 | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_LOG1P | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_LOG1P | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_SQRT | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_SQRT | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_SQRT | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_SQRT | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_CEIL | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_CEIL | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_TRUNC | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_TRUNC | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_FLOOR | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_FLOOR | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_RINT | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_RINT | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_MOD | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MOD | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_MOD | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_MOD | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_MOD | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_MOD | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_MOD | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_MOD | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_MOD | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_MOD | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_ISNAN | (BH_BOOL << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ADD_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_ADD_REDUCE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ADD_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_ADD_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_ADD_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_ADD_REDUCE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ADD_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_ADD_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_ADD_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_ADD_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_ADD_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_ADD_REDUCE | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_ADD_REDUCE | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_MULTIPLY_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_MULTIPLY_REDUCE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_MULTIPLY_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_MULTIPLY_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_MULTIPLY_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_MULTIPLY_REDUCE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_MULTIPLY_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_MULTIPLY_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_MULTIPLY_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_MULTIPLY_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_MULTIPLY_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_MULTIPLY_REDUCE | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_MULTIPLY_REDUCE | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_MINIMUM_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_MINIMUM_REDUCE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_MINIMUM_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_MINIMUM_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_MINIMUM_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_MINIMUM_REDUCE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_MINIMUM_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_MINIMUM_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_MINIMUM_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_MINIMUM_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_MINIMUM_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_MAXIMUM_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_MAXIMUM_REDUCE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_MAXIMUM_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_MAXIMUM_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_MAXIMUM_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_MAXIMUM_REDUCE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_MAXIMUM_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_MAXIMUM_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_MAXIMUM_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_MAXIMUM_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_MAXIMUM_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_LOGICAL_AND_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_LOGICAL_OR_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_LOGICAL_XOR_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
    }
    
    // The output type does not exist for the opcode
    if (desired_input_type1 == BH_UNKNOWN)
        return false;
    
    if (bh_operands(opcode) == 3)
    {
        // The output type does not exist for the opcode
        if (desired_input_type2 == BH_UNKNOWN)
            return false;

        if (*inputtype1 == BH_UNKNOWN)                      // First operand constant
        {
        // Check if we can convert with IDENTITY
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *constanttype, BH_UNKNOWN, BH_UNKNOWN) && bh_validate_types(BH_IDENTITY, desired_input_type2, *inputtype2, BH_UNKNOWN, BH_UNKNOWN))
            {
                *constanttype = desired_input_type1;
                *inputtype2 = desired_input_type2;
                return true;
            }
        }
        else if (*inputtype2 == BH_UNKNOWN)                 // Second operand constant
        {
            // Check if we can convert with IDENTITY
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *inputtype1, BH_UNKNOWN, BH_UNKNOWN) && bh_validate_types(BH_IDENTITY, desired_input_type2, *constanttype, BH_UNKNOWN, BH_UNKNOWN))
            {
                *inputtype1 = desired_input_type1;
                *constanttype = desired_input_type2;
                return true;
            }
        }
        else                                                // No constant
        {
            // Check if we can convert with IDENTITY
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *inputtype1, BH_UNKNOWN, BH_UNKNOWN) && bh_validate_types(BH_IDENTITY, desired_input_type2, *inputtype2, BH_UNKNOWN, BH_UNKNOWN))
            {
                *inputtype1 = desired_input_type1;
                *inputtype2 = desired_input_type2;
                return true;
            }
        }
    }
    else
    {
        // Check if we can convert with IDENTITY
        if (*inputtype1 == BH_UNKNOWN)                      // Constant input
        {
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *constanttype, BH_UNKNOWN, BH_UNKNOWN))
            {
                *constanttype = desired_input_type1;
                return true;
            }
        }
        else                                                // No constant
        {
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *inputtype1, BH_UNKNOWN, BH_UNKNOWN))
            {
                *inputtype1 = desired_input_type1;
                return true;
            }
        }
    }
    
    // Not possible to convert the types automatically
    return false;
}