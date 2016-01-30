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
#include <map>
#include <string>
#include <algorithm>
#include <tuple>
#include <iostream>
#include <sstream>

using namespace std;

static string const2str(const bh_constant &constant)
{
    stringstream ss;
    switch(constant.type)
    {
        case BH_BOOL:
            ss << constant.value.bool8;
            break;
        case BH_INT8:
            ss << constant.value.int8;
            break;
        case BH_INT16:
            ss << constant.value.int16;
            break;
        case BH_INT32:
            ss << constant.value.int32;
            break;
        case BH_INT64:
            ss << constant.value.int64;
            break;
        case BH_UINT8:
            ss << constant.value.uint8;
            break;
        case BH_UINT16:
            ss << constant.value.uint16;
            break;
        case BH_UINT32:
            ss << constant.value.uint32;
            break;
        case BH_UINT64:
            ss << constant.value.uint64;
            break;
        case BH_FLOAT32:
            ss << constant.value.float32;
            break;
        case BH_FLOAT64:
            ss << constant.value.float64;
            break;
        case BH_R123:
            ss << "{start: " << constant.value.r123.start << ", key: "\
               << constant.value.r123.key << "}";
            break;
        case BH_COMPLEX64:
            ss << constant.value.complex64.real << "+" \
               << constant.value.complex64.imag << "i";
            break;
        case BH_COMPLEX128:
            ss << constant.value.complex128.real << "+" \
               << constant.value.complex128.imag << "i";
            break;
        case BH_UNKNOWN:
        default:
            ss << "?";
    }
    return ss.str();
}

//Implements pprint of an instruction
ostream& operator<<(ostream& out, const bh_instruction& instr)
{
    string name;
    if(instr.opcode > BH_MAX_OPCODE_ID)//It is an extension method
        name = "ExtMethod";
    else//Regular instruction
        name = bh_opcode_text(instr.opcode);
    out << name;

    for(int i=0; i < bh_operands(instr.opcode); i++)
    {
        const bh_view &v = instr.operand[i];
        out << " ";
        if(bh_is_constant(&v))
            out << const2str(instr.constant);
        else
            out << v;
    }
    return out;
}

