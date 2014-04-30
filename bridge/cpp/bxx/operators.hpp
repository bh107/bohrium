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

#ifndef __BOHRIUM_BRIDGE_CPP_OPERATORS
#define __BOHRIUM_BRIDGE_CPP_OPERATORS
#include "bh.h"

namespace bxx {


//
//  Internally defined operator overloads
//

template <typename T>
inline multi_array<T>& multi_array<T>::operator+= (const T& rhs)
{
    return bh_add (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator+= (multi_array<T>& rhs)
{
    return bh_add (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator-= (const T& rhs)
{
    return bh_subtract (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator-= (multi_array<T>& rhs)
{
    return bh_subtract (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator*= (const T& rhs)
{
    return bh_multiply (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator*= (multi_array<T>& rhs)
{
    return bh_multiply (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator/= (const T& rhs)
{
    return bh_divide (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator/= (multi_array<T>& rhs)
{
    return bh_divide (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator%= (const T& rhs)
{
    return bh_mod (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator%= (multi_array<T>& rhs)
{
    return bh_mod (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator&= (const T& rhs)
{
    return bh_bitwise_and (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator&= (multi_array<T>& rhs)
{
    return bh_bitwise_and (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator|= (const T& rhs)
{
    return bh_bitwise_or (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator|= (multi_array<T>& rhs)
{
    return bh_bitwise_or (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator^= (const T& rhs)
{
    return bh_bitwise_xor (*this, *this, rhs);
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator^= (multi_array<T>& rhs)
{
    return bh_bitwise_xor (*this, *this, rhs);
}


template <typename T>
inline multi_array<T>& add (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_add (lhs, rhs);
}

template <typename T>
inline multi_array<T>& add (multi_array<T>& lhs, const T rhs)
{
    return bh_add (lhs, rhs);
}

template <typename T>
inline multi_array<T>& add (const T lhs, multi_array<T>& rhs)
{
    return bh_add (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator+ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_add (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator+ (multi_array<T>& lhs, const T rhs)
{
    return bh_add (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator+ (const T lhs, multi_array<T>& rhs)
{
    return bh_add (lhs, rhs);
}

template <typename T>
inline multi_array<T>& subtract (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_subtract (lhs, rhs);
}

template <typename T>
inline multi_array<T>& subtract (multi_array<T>& lhs, const T rhs)
{
    return bh_subtract (lhs, rhs);
}

template <typename T>
inline multi_array<T>& subtract (const T lhs, multi_array<T>& rhs)
{
    return bh_subtract (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator- (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_subtract (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator- (multi_array<T>& lhs, const T rhs)
{
    return bh_subtract (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator- (const T lhs, multi_array<T>& rhs)
{
    return bh_subtract (lhs, rhs);
}

template <typename T>
inline multi_array<T>& mul (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_multiply (lhs, rhs);
}

template <typename T>
inline multi_array<T>& mul (multi_array<T>& lhs, const T rhs)
{
    return bh_multiply (lhs, rhs);
}

template <typename T>
inline multi_array<T>& mul (const T lhs, multi_array<T>& rhs)
{
    return bh_multiply (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator* (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_multiply (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator* (multi_array<T>& lhs, const T rhs)
{
    return bh_multiply (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator* (const T lhs, multi_array<T>& rhs)
{
    return bh_multiply (lhs, rhs);
}

template <typename T>
inline multi_array<T>& div (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_divide (lhs, rhs);
}

template <typename T>
inline multi_array<T>& div (multi_array<T>& lhs, const T rhs)
{
    return bh_divide (lhs, rhs);
}

template <typename T>
inline multi_array<T>& div (const T lhs, multi_array<T>& rhs)
{
    return bh_divide (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator/ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_divide (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator/ (multi_array<T>& lhs, const T rhs)
{
    return bh_divide (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator/ (const T lhs, multi_array<T>& rhs)
{
    return bh_divide (lhs, rhs);
}

template <typename T>
inline multi_array<T>& mod (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_mod (lhs, rhs);
}

template <typename T>
inline multi_array<T>& mod (multi_array<T>& lhs, const T rhs)
{
    return bh_mod (lhs, rhs);
}

template <typename T>
inline multi_array<T>& mod (const T lhs, multi_array<T>& rhs)
{
    return bh_mod (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator% (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_mod (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator% (multi_array<T>& lhs, const T rhs)
{
    return bh_mod (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator% (const T lhs, multi_array<T>& rhs)
{
    return bh_mod (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_and (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_bitwise_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_and (multi_array<T>& lhs, const T rhs)
{
    return bh_bitwise_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_and (const T lhs, multi_array<T>& rhs)
{
    return bh_bitwise_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator& (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_bitwise_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator& (multi_array<T>& lhs, const T rhs)
{
    return bh_bitwise_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator& (const T lhs, multi_array<T>& rhs)
{
    return bh_bitwise_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_or (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_bitwise_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_or (multi_array<T>& lhs, const T rhs)
{
    return bh_bitwise_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_or (const T lhs, multi_array<T>& rhs)
{
    return bh_bitwise_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator| (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_bitwise_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator| (multi_array<T>& lhs, const T rhs)
{
    return bh_bitwise_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator| (const T lhs, multi_array<T>& rhs)
{
    return bh_bitwise_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_xor (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_bitwise_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_xor (multi_array<T>& lhs, const T rhs)
{
    return bh_bitwise_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& bitwise_xor (const T lhs, multi_array<T>& rhs)
{
    return bh_bitwise_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator^ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_bitwise_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator^ (multi_array<T>& lhs, const T rhs)
{
    return bh_bitwise_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator^ (const T lhs, multi_array<T>& rhs)
{
    return bh_bitwise_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& left_shift (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_left_shift (lhs, rhs);
}

template <typename T>
inline multi_array<T>& left_shift (multi_array<T>& lhs, const T rhs)
{
    return bh_left_shift (lhs, rhs);
}

template <typename T>
inline multi_array<T>& left_shift (const T lhs, multi_array<T>& rhs)
{
    return bh_left_shift (lhs, rhs);
}

template <typename T>
inline multi_array<T>& right_shift (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_right_shift (lhs, rhs);
}

template <typename T>
inline multi_array<T>& right_shift (multi_array<T>& lhs, const T rhs)
{
    return bh_right_shift (lhs, rhs);
}

template <typename T>
inline multi_array<T>& right_shift (const T lhs, multi_array<T>& rhs)
{
    return bh_right_shift (lhs, rhs);
}

template <typename T>
inline multi_array<T>& eq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& eq (multi_array<T>& lhs, const T rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& eq (const T lhs, multi_array<T>& rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator== (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator== (multi_array<T>& lhs, const T rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator== (const T lhs, multi_array<T>& rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& neq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& neq (multi_array<T>& lhs, const T rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& neq (const T lhs, multi_array<T>& rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator!= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator!= (multi_array<T>& lhs, const T rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator!= (const T lhs, multi_array<T>& rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& gt (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<T>& gt (multi_array<T>& lhs, const T rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<T>& gt (const T lhs, multi_array<T>& rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator> (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator> (multi_array<T>& lhs, const T rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator> (const T lhs, multi_array<T>& rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<T>& gteq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& gteq (multi_array<T>& lhs, const T rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& gteq (const T lhs, multi_array<T>& rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator>= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator>= (multi_array<T>& lhs, const T rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator>= (const T lhs, multi_array<T>& rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& lt (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<T>& lt (multi_array<T>& lhs, const T rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<T>& lt (const T lhs, multi_array<T>& rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator< (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator< (multi_array<T>& lhs, const T rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator< (const T lhs, multi_array<T>& rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<T>& lteq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& lteq (multi_array<T>& lhs, const T rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& lteq (const T lhs, multi_array<T>& rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator<= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator<= (multi_array<T>& lhs, const T rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator<= (const T lhs, multi_array<T>& rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_and (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_logical_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_and (multi_array<T>& lhs, const T rhs)
{
    return bh_logical_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_and (const T lhs, multi_array<T>& rhs)
{
    return bh_logical_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator&& (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_logical_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator&& (multi_array<T>& lhs, const T rhs)
{
    return bh_logical_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator&& (const T lhs, multi_array<T>& rhs)
{
    return bh_logical_and (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_or (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_logical_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_or (multi_array<T>& lhs, const T rhs)
{
    return bh_logical_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_or (const T lhs, multi_array<T>& rhs)
{
    return bh_logical_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator|| (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_logical_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator|| (multi_array<T>& lhs, const T rhs)
{
    return bh_logical_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& operator|| (const T lhs, multi_array<T>& rhs)
{
    return bh_logical_or (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_xor (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_logical_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_xor (multi_array<T>& lhs, const T rhs)
{
    return bh_logical_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& logical_xor (const T lhs, multi_array<T>& rhs)
{
    return bh_logical_xor (lhs, rhs);
}

template <typename T>
inline multi_array<T>& pow (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_power (lhs, rhs);
}

template <typename T>
inline multi_array<T>& pow (multi_array<T>& lhs, const T rhs)
{
    return bh_power (lhs, rhs);
}

template <typename T>
inline multi_array<T>& pow (const T lhs, multi_array<T>& rhs)
{
    return bh_power (lhs, rhs);
}



template <typename OutT, typename InT>
inline
multi_array<OutT>& logical_not (multi_array<InT>& rhs)
{
    return bh_logical_not (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& logical_not (const InT rhs)
{
    return bh_logical_not (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& operator! (multi_array<InT>& rhs)
{
    return bh_logical_not (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& operator! (const InT rhs)
{
    return bh_logical_not (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& invert (multi_array<InT>& rhs)
{
    return bh_invert (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& invert (const InT rhs)
{
    return bh_invert (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& imag (multi_array<InT>& rhs)
{
    return bh_imag (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& imag (const InT rhs)
{
    return bh_imag (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& real (multi_array<InT>& rhs)
{
    return bh_real (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& real (const InT rhs)
{
    return bh_real (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& abs (multi_array<InT>& rhs)
{
    return bh_absolute (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& abs (const InT rhs)
{
    return bh_absolute (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& max (multi_array<InT>& rhs)
{
    return bh_maximum (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& max (const InT rhs)
{
    return bh_maximum (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& min (multi_array<InT>& rhs)
{
    return bh_minimum (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& min (const InT rhs)
{
    return bh_minimum (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& sin (multi_array<InT>& rhs)
{
    return bh_sin (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& sin (const InT rhs)
{
    return bh_sin (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& cos (multi_array<InT>& rhs)
{
    return bh_cos (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& cos (const InT rhs)
{
    return bh_cos (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& tan (multi_array<InT>& rhs)
{
    return bh_tan (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& tan (const InT rhs)
{
    return bh_tan (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& sinh (multi_array<InT>& rhs)
{
    return bh_sinh (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& sinh (const InT rhs)
{
    return bh_sinh (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& cosh (multi_array<InT>& rhs)
{
    return bh_cosh (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& cosh (const InT rhs)
{
    return bh_cosh (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& tanh (multi_array<InT>& rhs)
{
    return bh_tanh (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& tanh (const InT rhs)
{
    return bh_tanh (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& asin (multi_array<InT>& rhs)
{
    return bh_arcsin (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& asin (const InT rhs)
{
    return bh_arcsin (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& acos (multi_array<InT>& rhs)
{
    return bh_arccos (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& acos (const InT rhs)
{
    return bh_arccos (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& atan (multi_array<InT>& rhs)
{
    return bh_arctan (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& atan (const InT rhs)
{
    return bh_arctan (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& atan2 (multi_array<InT>& rhs)
{
    return bh_arctan2 (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& atan2 (const InT rhs)
{
    return bh_arctan2 (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& asinh (multi_array<InT>& rhs)
{
    return bh_arcsinh (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& asinh (const InT rhs)
{
    return bh_arcsinh (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& acosh (multi_array<InT>& rhs)
{
    return bh_arccosh (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& acosh (const InT rhs)
{
    return bh_arccosh (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& atanh (multi_array<InT>& rhs)
{
    return bh_arctanh (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& atanh (const InT rhs)
{
    return bh_arctanh (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& exp (multi_array<InT>& rhs)
{
    return bh_exp (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& exp (const InT rhs)
{
    return bh_exp (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& exp2 (multi_array<InT>& rhs)
{
    return bh_exp2 (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& exp2 (const InT rhs)
{
    return bh_exp2 (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& expm1 (multi_array<InT>& rhs)
{
    return bh_expm1 (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& expm1 (const InT rhs)
{
    return bh_expm1 (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& isnan (multi_array<InT>& rhs)
{
    return bh_isnan (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& isnan (const InT rhs)
{
    return bh_isnan (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& isinf (multi_array<InT>& rhs)
{
    return bh_isinf (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& isinf (const InT rhs)
{
    return bh_isinf (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& log (multi_array<InT>& rhs)
{
    return bh_log (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& log (const InT rhs)
{
    return bh_log (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& log2 (multi_array<InT>& rhs)
{
    return bh_log2 (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& log2 (const InT rhs)
{
    return bh_log2 (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& log10 (multi_array<InT>& rhs)
{
    return bh_log10 (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& log10 (const InT rhs)
{
    return bh_log10 (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& log1p (multi_array<InT>& rhs)
{
    return bh_log1p (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& log1p (const InT rhs)
{
    return bh_log1p (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& sqrt (multi_array<InT>& rhs)
{
    return bh_sqrt (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& sqrt (const InT rhs)
{
    return bh_sqrt (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& ceil (multi_array<InT>& rhs)
{
    return bh_ceil (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& ceil (const InT rhs)
{
    return bh_ceil (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& trunc (multi_array<InT>& rhs)
{
    return bh_trunc (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& trunc (const InT rhs)
{
    return bh_trunc (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& floor (multi_array<InT>& rhs)
{
    return bh_floor (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& floor (const InT rhs)
{
    return bh_floor (rhs);
}


template <typename OutT, typename InT>
inline
multi_array<OutT>& rint (multi_array<InT>& rhs)
{
    return bh_rint (rhs);
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& rint (const InT rhs)
{
    return bh_rint (rhs);
}


}
#endif
