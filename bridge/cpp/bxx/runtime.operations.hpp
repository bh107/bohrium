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

//
//  WARN:   This file is generated; changes to it will be overwritten.
//          If you wish to change its functionality then change the code-generator for this file.
//          Take a look at: codegen/README
//

#ifndef __BOHRIUM_BRIDGE_CPP_RUNTIME_OPERATIONS
#define __BOHRIUM_BRIDGE_CPP_RUNTIME_OPERATIONS
#include <bh.h>

namespace bxx {



// bh_none - BH_NONE - runtime.nops0 - 0 ()
inline
void bh_none (void)
{
    Runtime::instance().enqueue((bh_opcode)BH_NONE);
}

// bh_tally - BH_TALLY - runtime.nops0 - 0 ()
inline
void bh_tally (void)
{
    Runtime::instance().enqueue((bh_opcode)BH_TALLY);
}


// bh_add - BH_ADD - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_add (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ADD, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, rhs);
    return res;
}

// bh_add - BH_ADD - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_add (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_ADD, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, rhs);
    return res;
}

// bh_add - BH_ADD - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_add (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ADD, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, rhs);
    return res;
}

// bh_subtract - BH_SUBTRACT - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_subtract (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, rhs);
    return res;
}

// bh_subtract - BH_SUBTRACT - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_subtract (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, rhs);
    return res;
}

// bh_subtract - BH_SUBTRACT - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_subtract (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, rhs);
    return res;
}

// bh_multiply - BH_MULTIPLY - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_multiply (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, rhs);
    return res;
}

// bh_multiply - BH_MULTIPLY - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_multiply (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, rhs);
    return res;
}

// bh_multiply - BH_MULTIPLY - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_multiply (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, rhs);
    return res;
}

// bh_divide - BH_DIVIDE - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_divide (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, rhs);
    return res;
}

// bh_divide - BH_DIVIDE - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_divide (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, rhs);
    return res;
}

// bh_divide - BH_DIVIDE - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_divide (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, rhs);
    return res;
}

// bh_mod - BH_MOD - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_mod (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MOD, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, rhs);
    return res;
}

// bh_mod - BH_MOD - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_mod (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MOD, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, rhs);
    return res;
}

// bh_mod - BH_MOD - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_mod (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MOD, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, rhs);
    return res;
}

// bh_bitwise_and - BH_BITWISE_AND - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_and (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, rhs);
    return res;
}

// bh_bitwise_and - BH_BITWISE_AND - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_and (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, rhs);
    return res;
}

// bh_bitwise_and - BH_BITWISE_AND - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_and (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, rhs);
    return res;
}

// bh_bitwise_or - BH_BITWISE_OR - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_or (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, rhs);
    return res;
}

// bh_bitwise_or - BH_BITWISE_OR - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_or (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, rhs);
    return res;
}

// bh_bitwise_or - BH_BITWISE_OR - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_or (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, rhs);
    return res;
}

// bh_bitwise_xor - BH_BITWISE_XOR - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_xor (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, rhs);
    return res;
}

// bh_bitwise_xor - BH_BITWISE_XOR - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_xor (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, rhs);
    return res;
}

// bh_bitwise_xor - BH_BITWISE_XOR - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_bitwise_xor (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, rhs);
    return res;
}

// bh_left_shift - BH_LEFT_SHIFT - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_left_shift (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, rhs);
    return res;
}

// bh_left_shift - BH_LEFT_SHIFT - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_left_shift (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, rhs);
    return res;
}

// bh_left_shift - BH_LEFT_SHIFT - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_left_shift (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, rhs);
    return res;
}

// bh_right_shift - BH_RIGHT_SHIFT - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_right_shift (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, rhs);
    return res;
}

// bh_right_shift - BH_RIGHT_SHIFT - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_right_shift (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, rhs);
    return res;
}

// bh_right_shift - BH_RIGHT_SHIFT - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_right_shift (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, rhs);
    return res;
}

// bh_equal - BH_EQUAL - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_equal (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, rhs);
    return res;
}

// bh_equal - BH_EQUAL - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_equal (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, rhs);
    return res;
}

// bh_equal - BH_EQUAL - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_equal (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, rhs);
    return res;
}

// bh_not_equal - BH_NOT_EQUAL - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_not_equal (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_NOT_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, rhs);
    return res;
}

// bh_not_equal - BH_NOT_EQUAL - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_not_equal (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_NOT_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, rhs);
    return res;
}

// bh_not_equal - BH_NOT_EQUAL - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_not_equal (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_NOT_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, rhs);
    return res;
}

// bh_greater - BH_GREATER - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_greater (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, rhs);
    return res;
}

// bh_greater - BH_GREATER - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_greater (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_GREATER, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, rhs);
    return res;
}

// bh_greater - BH_GREATER - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_greater (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, rhs);
    return res;
}

// bh_greater_equal - BH_GREATER_EQUAL - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_greater_equal (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, rhs);
    return res;
}

// bh_greater_equal - BH_GREATER_EQUAL - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_greater_equal (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, rhs);
    return res;
}

// bh_greater_equal - BH_GREATER_EQUAL - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_greater_equal (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, rhs);
    return res;
}

// bh_less - BH_LESS - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_less (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LESS, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, rhs);
    return res;
}

// bh_less - BH_LESS - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_less (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LESS, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, rhs);
    return res;
}

// bh_less - BH_LESS - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_less (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LESS, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, rhs);
    return res;
}

// bh_less_equal - BH_LESS_EQUAL - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_less_equal (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, rhs);
    return res;
}

// bh_less_equal - BH_LESS_EQUAL - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_less_equal (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, rhs);
    return res;
}

// bh_less_equal - BH_LESS_EQUAL - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_less_equal (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, rhs);
    return res;
}

// bh_logical_and - BH_LOGICAL_AND - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_and (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, rhs);
    return res;
}

// bh_logical_and - BH_LOGICAL_AND - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_and (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, rhs);
    return res;
}

// bh_logical_and - BH_LOGICAL_AND - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_and (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, rhs);
    return res;
}

// bh_logical_or - BH_LOGICAL_OR - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_or (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, rhs);
    return res;
}

// bh_logical_or - BH_LOGICAL_OR - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_or (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, rhs);
    return res;
}

// bh_logical_or - BH_LOGICAL_OR - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_or (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, rhs);
    return res;
}

// bh_logical_xor - BH_LOGICAL_XOR - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_xor (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, rhs);
    return res;
}

// bh_logical_xor - BH_LOGICAL_XOR - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_xor (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, rhs);
    return res;
}

// bh_logical_xor - BH_LOGICAL_XOR - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_logical_xor (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, rhs);
    return res;
}

// bh_power - BH_POWER - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_power (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_POWER, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

// bh_power - BH_POWER - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_power (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_POWER, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

// bh_power - BH_POWER - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_power (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_POWER, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, rhs);
    return res;
}

// bh_maximum - BH_MAXIMUM - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_maximum (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

// bh_maximum - BH_MAXIMUM - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_maximum (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

// bh_maximum - BH_MAXIMUM - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_maximum (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, rhs);
    return res;
}

// bh_minimum - BH_MINIMUM - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_minimum (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

// bh_minimum - BH_MINIMUM - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_minimum (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

// bh_minimum - BH_MINIMUM - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_minimum (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, rhs);
    return res;
}

// bh_arctan2 - BH_ARCTAN2 - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_arctan2 (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, lhs, rhs);
    return res;
}

// bh_arctan2 - BH_ARCTAN2 - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_arctan2 (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, lhs, rhs);
    return res;
}

// bh_arctan2 - BH_ARCTAN2 - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_arctan2 (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, lhs, rhs);
    return res;
}

// bh_scatter - BH_SCATTER - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_scatter (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_SCATTER, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_SCATTER, res, lhs, rhs);
    return res;
}



// bh_gather - BH_GATHER - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_gather (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_GATHER, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_GATHER, res, lhs, rhs);
    return res;
}



// bh_matmul - BH_MATMUL - runtime.nops3 - 3 (A,A,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_matmul (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, res, lhs, rhs);
    return res;
}

// bh_matmul - BH_MATMUL - runtime.nops3 - 3 (A,A,K)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_matmul (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, res, lhs, rhs);
    return res;
}

// bh_matmul - BH_MATMUL - runtime.nops3 - 3 (A,K,A)
template <typename TO, typename TL, typename TR>
inline
multi_array<TO>& bh_matmul (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TO, TL, TR>();
    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, res, lhs, rhs);
    return res;
}


// bh_identity - BH_IDENTITY - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_identity (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_IDENTITY, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, res, rhs);
    return res;
}

// bh_identity - BH_IDENTITY - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_identity (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_IDENTITY, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, res, rhs);
    return res;
}

// bh_logical_not - BH_LOGICAL_NOT - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_logical_not (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_NOT, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, res, rhs);
    return res;
}

// bh_logical_not - BH_LOGICAL_NOT - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_logical_not (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_NOT, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, res, rhs);
    return res;
}

// bh_invert - BH_INVERT - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_invert (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_INVERT, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_INVERT, res, rhs);
    return res;
}

// bh_invert - BH_INVERT - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_invert (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_INVERT, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_INVERT, res, rhs);
    return res;
}

// bh_imag - BH_IMAG - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_imag (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_IMAG, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_IMAG, res, rhs);
    return res;
}

// bh_imag - BH_IMAG - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_imag (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_IMAG, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_IMAG, res, rhs);
    return res;
}

// bh_real - BH_REAL - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_real (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_REAL, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_REAL, res, rhs);
    return res;
}

// bh_real - BH_REAL - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_real (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_REAL, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_REAL, res, rhs);
    return res;
}

// bh_absolute - BH_ABSOLUTE - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_absolute (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ABSOLUTE, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, res, rhs);
    return res;
}

// bh_absolute - BH_ABSOLUTE - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_absolute (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ABSOLUTE, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, res, rhs);
    return res;
}

// bh_sin - BH_SIN - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sin (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_SIN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_SIN, res, rhs);
    return res;
}

// bh_sin - BH_SIN - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sin (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_SIN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_SIN, res, rhs);
    return res;
}

// bh_cos - BH_COS - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_cos (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_COS, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_COS, res, rhs);
    return res;
}

// bh_cos - BH_COS - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_cos (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_COS, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_COS, res, rhs);
    return res;
}

// bh_tan - BH_TAN - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_tan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_TAN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_TAN, res, rhs);
    return res;
}

// bh_tan - BH_TAN - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_tan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_TAN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_TAN, res, rhs);
    return res;
}

// bh_sinh - BH_SINH - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sinh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_SINH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_SINH, res, rhs);
    return res;
}

// bh_sinh - BH_SINH - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sinh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_SINH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_SINH, res, rhs);
    return res;
}

// bh_cosh - BH_COSH - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_cosh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_COSH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_COSH, res, rhs);
    return res;
}

// bh_cosh - BH_COSH - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_cosh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_COSH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_COSH, res, rhs);
    return res;
}

// bh_tanh - BH_TANH - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_tanh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_TANH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_TANH, res, rhs);
    return res;
}

// bh_tanh - BH_TANH - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_tanh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_TANH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_TANH, res, rhs);
    return res;
}

// bh_arcsin - BH_ARCSIN - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arcsin (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCSIN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, res, rhs);
    return res;
}

// bh_arcsin - BH_ARCSIN - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arcsin (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCSIN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, res, rhs);
    return res;
}

// bh_arccos - BH_ARCCOS - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arccos (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCCOS, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, res, rhs);
    return res;
}

// bh_arccos - BH_ARCCOS - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arccos (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCCOS, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, res, rhs);
    return res;
}

// bh_arctan - BH_ARCTAN - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arctan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, res, rhs);
    return res;
}

// bh_arctan - BH_ARCTAN - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arctan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, res, rhs);
    return res;
}

// bh_arcsinh - BH_ARCSINH - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arcsinh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCSINH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, res, rhs);
    return res;
}

// bh_arcsinh - BH_ARCSINH - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arcsinh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCSINH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, res, rhs);
    return res;
}

// bh_arccosh - BH_ARCCOSH - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arccosh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCCOSH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, res, rhs);
    return res;
}

// bh_arccosh - BH_ARCCOSH - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arccosh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCCOSH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, res, rhs);
    return res;
}

// bh_arctanh - BH_ARCTANH - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arctanh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCTANH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, res, rhs);
    return res;
}

// bh_arctanh - BH_ARCTANH - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arctanh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCTANH, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, res, rhs);
    return res;
}

// bh_exp - BH_EXP - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_exp (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_EXP, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_EXP, res, rhs);
    return res;
}

// bh_exp - BH_EXP - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_exp (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_EXP, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_EXP, res, rhs);
    return res;
}

// bh_exp2 - BH_EXP2 - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_exp2 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_EXP2, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_EXP2, res, rhs);
    return res;
}

// bh_exp2 - BH_EXP2 - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_exp2 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_EXP2, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_EXP2, res, rhs);
    return res;
}

// bh_expm1 - BH_EXPM1 - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_expm1 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_EXPM1, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, res, rhs);
    return res;
}

// bh_expm1 - BH_EXPM1 - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_expm1 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_EXPM1, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, res, rhs);
    return res;
}

// bh_isnan - BH_ISNAN - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_isnan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ISNAN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, res, rhs);
    return res;
}

// bh_isnan - BH_ISNAN - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_isnan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ISNAN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, res, rhs);
    return res;
}

// bh_isinf - BH_ISINF - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_isinf (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ISINF, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ISINF, res, rhs);
    return res;
}

// bh_isinf - BH_ISINF - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_isinf (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ISINF, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_ISINF, res, rhs);
    return res;
}

// bh_log - BH_LOG - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOG, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOG, res, rhs);
    return res;
}

// bh_log - BH_LOG - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOG, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOG, res, rhs);
    return res;
}

// bh_log2 - BH_LOG2 - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log2 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOG2, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOG2, res, rhs);
    return res;
}

// bh_log2 - BH_LOG2 - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log2 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOG2, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOG2, res, rhs);
    return res;
}

// bh_log10 - BH_LOG10 - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log10 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOG10, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOG10, res, rhs);
    return res;
}

// bh_log10 - BH_LOG10 - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log10 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOG10, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOG10, res, rhs);
    return res;
}

// bh_log1p - BH_LOG1P - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log1p (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOG1P, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, res, rhs);
    return res;
}

// bh_log1p - BH_LOG1P - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log1p (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOG1P, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, res, rhs);
    return res;
}

// bh_sqrt - BH_SQRT - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sqrt (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_SQRT, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_SQRT, res, rhs);
    return res;
}

// bh_sqrt - BH_SQRT - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sqrt (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_SQRT, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_SQRT, res, rhs);
    return res;
}

// bh_ceil - BH_CEIL - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_ceil (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_CEIL, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_CEIL, res, rhs);
    return res;
}

// bh_ceil - BH_CEIL - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_ceil (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_CEIL, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_CEIL, res, rhs);
    return res;
}

// bh_trunc - BH_TRUNC - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_trunc (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_TRUNC, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, res, rhs);
    return res;
}

// bh_trunc - BH_TRUNC - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_trunc (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_TRUNC, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, res, rhs);
    return res;
}

// bh_floor - BH_FLOOR - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_floor (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_FLOOR, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, res, rhs);
    return res;
}

// bh_floor - BH_FLOOR - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_floor (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_FLOOR, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, res, rhs);
    return res;
}

// bh_rint - BH_RINT - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_rint (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_RINT, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_RINT, res, rhs);
    return res;
}

// bh_rint - BH_RINT - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_rint (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_RINT, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_RINT, res, rhs);
    return res;
}

// bh_sign - BH_SIGN - runtime.nops2 - 2 (A,A)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sign (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_SIGN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_SIGN, res, rhs);
    return res;
}

// bh_sign - BH_SIGN - runtime.nops2 - 2 (A,K)
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sign (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_SIGN, OutT, InT>();
    Runtime::instance().enqueue((bh_opcode)BH_SIGN, res, rhs);
    return res;
}


// bh_range - BH_RANGE - runtime.nops1 - 1 (A)
template <typename T>
inline
multi_array<T>& bh_range (multi_array<T>& res)
{
    Runtime::instance().typecheck<BH_RANGE, T>();
    Runtime::instance().enqueue((bh_opcode)BH_RANGE, res);
    return res;
}

// bh_free - BH_FREE - runtime.nops1 - 1 (A)
template <typename T>
inline
multi_array<T>& bh_free (multi_array<T>& res)
{
    Runtime::instance().typecheck<BH_FREE, T>();
    Runtime::instance().enqueue((bh_opcode)BH_FREE, res);
    return res;
}

// bh_sync - BH_SYNC - runtime.nops1 - 1 (A)
template <typename T>
inline
multi_array<T>& bh_sync (multi_array<T>& res)
{
    Runtime::instance().typecheck<BH_SYNC, T>();
    Runtime::instance().enqueue((bh_opcode)BH_SYNC, res);
    return res;
}

// bh_discard - BH_DISCARD - runtime.nops1 - 1 (A)
template <typename T>
inline
multi_array<T>& bh_discard (multi_array<T>& res)
{
    Runtime::instance().typecheck<BH_DISCARD, T>();
    Runtime::instance().enqueue((bh_opcode)BH_DISCARD, res);
    return res;
}


template <typename T>
multi_array<T>& bh_random (multi_array<T>& res, uint64_t in1, uint64_t in2)
{
    Runtime::instance().typecheck<BH_RANDOM, T, uint64_t, uint64_t>();

    Runtime::instance().enqueue((bh_opcode)BH_RANDOM, res, in1, in2);

    return res;
}



template <typename T>
multi_array<T>& bh_add_accumulate (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    Runtime::instance().typecheck<BH_ADD_ACCUMULATE, T, T, int64_t>();

    // Check axis
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }
    // TODO:
    //  * shape-check
    //  * type-check
    Runtime::instance().enqueue((bh_opcode)BH_ADD_ACCUMULATE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_add_accumulate (multi_array<T> &lhs, int64_t rhs)
{
    Runtime::instance().typecheck<BH_ADD_ACCUMULATE, T, T, int64_t>();

    // Check axis
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ADD_ACCUMULATE, *result, lhs, rhs);
    result->setTemp(true);

    return *result;
}


template <typename T>
multi_array<T>& bh_multiply_accumulate (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY_ACCUMULATE, T, T, int64_t>();

    // Check axis
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }
    // TODO:
    //  * shape-check
    //  * type-check
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_ACCUMULATE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_multiply_accumulate (multi_array<T> &lhs, int64_t rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY_ACCUMULATE, T, T, int64_t>();

    // Check axis
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_ACCUMULATE, *result, lhs, rhs);
    result->setTemp(true);

    return *result;
}




//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_add_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_ADD_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_add_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_ADD_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ADD_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_multiply_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MULTIPLY_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_multiply_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MULTIPLY_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_minimum_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MINIMUM_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_minimum_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MINIMUM_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_maximum_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MAXIMUM_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_maximum_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MAXIMUM_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_logical_and_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_AND_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_logical_and_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_AND_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_logical_or_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_OR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_logical_or_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_OR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_logical_xor_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_XOR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_logical_xor_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_XOR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_bitwise_and_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_AND_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_bitwise_and_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_AND_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_bitwise_or_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_OR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_bitwise_or_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_OR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_bitwise_xor_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_XOR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_bitwise_xor_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_XOR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


}
#endif
