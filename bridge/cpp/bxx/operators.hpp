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
#include <bh.h>

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

template <typename T>
inline multi_array<T>& maximum (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_maximum (lhs, rhs);
}

template <typename T>
inline multi_array<T>& maximum (multi_array<T>& lhs, const T rhs)
{
    return bh_maximum (lhs, rhs);
}

template <typename T>
inline multi_array<T>& maximum (const T lhs, multi_array<T>& rhs)
{
    return bh_maximum (lhs, rhs);
}

template <typename T>
inline multi_array<T>& minimum (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_minimum (lhs, rhs);
}

template <typename T>
inline multi_array<T>& minimum (multi_array<T>& lhs, const T rhs)
{
    return bh_minimum (lhs, rhs);
}

template <typename T>
inline multi_array<T>& minimum (const T lhs, multi_array<T>& rhs)
{
    return bh_minimum (lhs, rhs);
}

template <typename T>
inline multi_array<T>& atan2 (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_arctan2 (lhs, rhs);
}

template <typename T>
inline multi_array<T>& atan2 (multi_array<T>& lhs, const T rhs)
{
    return bh_arctan2 (lhs, rhs);
}

template <typename T>
inline multi_array<T>& atan2 (const T lhs, multi_array<T>& rhs)
{
    return bh_arctan2 (lhs, rhs);
}



template <typename T>
inline multi_array<bool>& eq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& eq (multi_array<T>& lhs, const T rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& eq (const T lhs, multi_array<T>& rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator== (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator== (multi_array<T>& lhs, const T rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator== (const T lhs, multi_array<T>& rhs)
{
    return bh_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& neq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& neq (multi_array<T>& lhs, const T rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& neq (const T lhs, multi_array<T>& rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator!= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator!= (multi_array<T>& lhs, const T rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator!= (const T lhs, multi_array<T>& rhs)
{
    return bh_not_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& gt (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& gt (multi_array<T>& lhs, const T rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& gt (const T lhs, multi_array<T>& rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator> (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator> (multi_array<T>& lhs, const T rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator> (const T lhs, multi_array<T>& rhs)
{
    return bh_greater (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& gteq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& gteq (multi_array<T>& lhs, const T rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& gteq (const T lhs, multi_array<T>& rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator>= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator>= (multi_array<T>& lhs, const T rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator>= (const T lhs, multi_array<T>& rhs)
{
    return bh_greater_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& lt (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& lt (multi_array<T>& lhs, const T rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& lt (const T lhs, multi_array<T>& rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator< (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator< (multi_array<T>& lhs, const T rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator< (const T lhs, multi_array<T>& rhs)
{
    return bh_less (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& lteq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& lteq (multi_array<T>& lhs, const T rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& lteq (const T lhs, multi_array<T>& rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator<= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator<= (multi_array<T>& lhs, const T rhs)
{
    return bh_less_equal (lhs, rhs);
}

template <typename T>
inline multi_array<bool>& operator<= (const T lhs, multi_array<T>& rhs)
{
    return bh_less_equal (lhs, rhs);
}



template <typename T>
inline
multi_array<T>& logical_not (multi_array<T>& rhs)
{
    return bh_logical_not (rhs);
}


template <typename T>
inline
multi_array<T>& operator! (multi_array<T>& rhs)
{
    return bh_logical_not (rhs);
}


template <typename T>
inline
multi_array<T>& invert (multi_array<T>& rhs)
{
    return bh_invert (rhs);
}


template <typename T>
inline
multi_array<T>& operator~ (multi_array<T>& rhs)
{
    return bh_invert (rhs);
}


template <typename T>
inline
multi_array<T>& abs (multi_array<T>& rhs)
{
    return bh_absolute (rhs);
}


template <typename T>
inline
multi_array<T>& sin (multi_array<T>& rhs)
{
    return bh_sin (rhs);
}


template <typename T>
inline
multi_array<T>& cos (multi_array<T>& rhs)
{
    return bh_cos (rhs);
}


template <typename T>
inline
multi_array<T>& tan (multi_array<T>& rhs)
{
    return bh_tan (rhs);
}


template <typename T>
inline
multi_array<T>& sinh (multi_array<T>& rhs)
{
    return bh_sinh (rhs);
}


template <typename T>
inline
multi_array<T>& cosh (multi_array<T>& rhs)
{
    return bh_cosh (rhs);
}


template <typename T>
inline
multi_array<T>& tanh (multi_array<T>& rhs)
{
    return bh_tanh (rhs);
}


template <typename T>
inline
multi_array<T>& asin (multi_array<T>& rhs)
{
    return bh_arcsin (rhs);
}


template <typename T>
inline
multi_array<T>& acos (multi_array<T>& rhs)
{
    return bh_arccos (rhs);
}


template <typename T>
inline
multi_array<T>& atan (multi_array<T>& rhs)
{
    return bh_arctan (rhs);
}


template <typename T>
inline
multi_array<T>& asinh (multi_array<T>& rhs)
{
    return bh_arcsinh (rhs);
}


template <typename T>
inline
multi_array<T>& acosh (multi_array<T>& rhs)
{
    return bh_arccosh (rhs);
}


template <typename T>
inline
multi_array<T>& atanh (multi_array<T>& rhs)
{
    return bh_arctanh (rhs);
}


template <typename T>
inline
multi_array<T>& exp (multi_array<T>& rhs)
{
    return bh_exp (rhs);
}


template <typename T>
inline
multi_array<T>& exp2 (multi_array<T>& rhs)
{
    return bh_exp2 (rhs);
}


template <typename T>
inline
multi_array<T>& expm1 (multi_array<T>& rhs)
{
    return bh_expm1 (rhs);
}


template <typename T>
inline
multi_array<T>& log (multi_array<T>& rhs)
{
    return bh_log (rhs);
}


template <typename T>
inline
multi_array<T>& log2 (multi_array<T>& rhs)
{
    return bh_log2 (rhs);
}


template <typename T>
inline
multi_array<T>& log10 (multi_array<T>& rhs)
{
    return bh_log10 (rhs);
}


template <typename T>
inline
multi_array<T>& log1p (multi_array<T>& rhs)
{
    return bh_log1p (rhs);
}


template <typename T>
inline
multi_array<T>& sqrt (multi_array<T>& rhs)
{
    return bh_sqrt (rhs);
}


template <typename T>
inline
multi_array<T>& ceil (multi_array<T>& rhs)
{
    return bh_ceil (rhs);
}


template <typename T>
inline
multi_array<T>& trunc (multi_array<T>& rhs)
{
    return bh_trunc (rhs);
}


template <typename T>
inline
multi_array<T>& floor (multi_array<T>& rhs)
{
    return bh_floor (rhs);
}


template <typename T>
inline
multi_array<T>& rint (multi_array<T>& rhs)
{
    return bh_rint (rhs);
}




template <typename T>
inline
multi_array<bool>& isnan (multi_array<T>& rhs)
{
    return bh_isnan (rhs);
}


template <typename T>
inline
multi_array<bool>& isinf (multi_array<T>& rhs)
{
    return bh_isinf (rhs);
}


}
#endif
