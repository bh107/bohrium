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

#ifndef __BOHRIUM_BRIDGE_CPP_FUNCTIONS
#define __BOHRIUM_BRIDGE_CPP_FUNCTIONS
#include "bh.h"

namespace bh {

template <typename T>
multi_array<T>& logical_xor (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& logical_xor (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& logical_xor (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& left_shift (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& left_shift (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& left_shift (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& right_shift (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& right_shift (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& right_shift (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& pow (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& pow (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& pow (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& abs (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& max (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& max (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& max (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& min (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& min (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& min (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& sin (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SIN, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& cos (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_COS, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& tan (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TAN, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& sinh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SINH, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& cosh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_COSH, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& tanh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TANH, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& asin (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& acos (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& atan (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& atan2 (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& atan2 (multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& atan2 (const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& asinh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& acosh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& atanh (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& exp (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXP, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& exp2 (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXP2, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& expm1 (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& isnan (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& isinf (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ISINF, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& log (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& log2 (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG2, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& log10 (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG10, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& log1p (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& sqrt (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SQRT, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& ceil (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_CEIL, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& trunc (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& floor (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, *result, rhs);
    return *result;
}
template <typename T>
multi_array<T>& rint (multi_array<T> &rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RINT, *result, rhs);
    return *result;
}

// TODO: Fix this hack-slash support for BH_REAL/IMAG
template <typename InT, typename OutT>
multi_array<OutT>& real (multi_array<InT> &rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT,InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_REAL, *result, rhs);
    return *result;
}
template <typename InT, typename OutT>
multi_array<OutT>& imag (multi_array<InT> &rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT,InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IMAG, *result, rhs);
    return *result;
}

}
#endif

