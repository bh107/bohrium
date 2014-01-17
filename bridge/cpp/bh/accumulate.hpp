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
#ifndef __BOHRIUM_BRIDGE_CPP_ACCUMULATE
#define __BOHRIUM_BRIDGE_CPP_ACCUMULATE

namespace bh {

inline bh_opcode accumulable_to_opcode(accumulable opcode)
{
    switch(opcode) {
        case SUM:
            return BH_ADD_ACCUMULATE;
            break;
        case PRODUCT:
            return BH_MULTIPLY_ACCUMULATE;
            break;
        default:
            throw std::runtime_error("Error: Unsupported opcode for accumulate.\n");
    }
}

template <typename T>
multi_array<T>& accumulate(multi_array<T>& op, accumulable opcode, size_t axis)
{
    multi_array<T>* result = &Runtime::instance().temp<T>(op);
    result->link();                         // Bind the base

    Runtime::instance().enqueue(accumulable_to_opcode(opcode), *result, op, (bh_int64)axis);

    return *result;
}

}
#endif

