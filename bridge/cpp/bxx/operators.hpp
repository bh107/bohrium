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



template <typename T>
inline
multi_array<T>& logical_not (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_logical_not (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& operator! (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_logical_not (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& invert (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_invert (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& operator~ (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_invert (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& abs (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_absolute (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& sin (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_sin (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& cos (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_cos (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& tan (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_tan (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& sinh (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_sinh (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& cosh (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_cosh (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& tanh (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_tanh (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& asin (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_arcsin (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& acos (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_arccos (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& atan (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_arctan (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& asinh (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_arcsinh (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& acosh (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_arccosh (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& atanh (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_arctanh (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& exp (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_exp (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& exp2 (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_exp2 (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& expm1 (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_expm1 (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& log (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_log (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& log2 (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_log2 (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& log10 (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_log10 (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& log1p (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_log1p (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& sqrt (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_sqrt (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& ceil (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_ceil (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& trunc (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_trunc (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& floor (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_floor (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& rint (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_rint (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<T>& sign (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    bh_sign (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}




template <typename T>
inline
multi_array<bool>& isnan (multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_isnan (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


template <typename T>
inline
multi_array<bool>& isinf (multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_isinf (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}




template <typename TL, typename TR>
inline multi_array<TL>& add (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_add (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& add (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_add (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& add (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_add (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator+ (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_add (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator+ (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_add (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator+ (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_add (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& subtract (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_subtract (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& subtract (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_subtract (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& subtract (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_subtract (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator- (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_subtract (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator- (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_subtract (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator- (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_subtract (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& mul (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_multiply (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& mul (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_multiply (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& mul (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_multiply (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator* (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_multiply (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator* (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_multiply (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator* (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_multiply (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& div (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_divide (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& div (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_divide (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& div (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_divide (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator/ (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_divide (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator/ (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_divide (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator/ (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_divide (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& mod (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_mod (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& mod (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_mod (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& mod (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_mod (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator% (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_mod (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator% (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_mod (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator% (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_mod (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_and (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_bitwise_and (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_and (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_bitwise_and (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_and (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_bitwise_and (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator& (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_bitwise_and (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator& (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_bitwise_and (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator& (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_bitwise_and (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_or (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_bitwise_or (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_or (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_bitwise_or (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_or (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_bitwise_or (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator| (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_bitwise_or (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator| (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_bitwise_or (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator| (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_bitwise_or (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_xor (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_bitwise_xor (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_xor (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_bitwise_xor (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& bitwise_xor (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_bitwise_xor (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator^ (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_bitwise_xor (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator^ (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_bitwise_xor (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator^ (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_bitwise_xor (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& left_shift (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_left_shift (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& left_shift (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_left_shift (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& left_shift (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_left_shift (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& right_shift (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_right_shift (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& right_shift (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_right_shift (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& right_shift (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_right_shift (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_and (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_logical_and (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_and (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_logical_and (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_and (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_logical_and (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator&& (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_logical_and (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator&& (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_logical_and (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator&& (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_logical_and (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_or (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_logical_or (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_or (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_logical_or (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_or (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_logical_or (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator|| (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_logical_or (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator|| (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_logical_or (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& operator|| (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_logical_or (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_xor (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_logical_xor (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_xor (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_logical_xor (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& logical_xor (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_logical_xor (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& pow (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_power (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& pow (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_power (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& pow (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_power (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& maximum (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_maximum (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& maximum (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_maximum (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& maximum (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_maximum (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& minimum (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_minimum (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& minimum (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_minimum (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& minimum (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_minimum (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& atan2 (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_arctan2 (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& atan2 (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_arctan2 (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& atan2 (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_arctan2 (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& matmul (multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(*left); // Construct result
    bh_matmul (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& matmul (multi_array<TL>& lhs, const TR rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TL>(lhs); // Construct result
    bh_matmul (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename TL, typename TR>
inline multi_array<TL>& matmul (const TL lhs, multi_array<TR>& rhs)
{
    multi_array<TL>* res = &Runtime::instance().create_base<TL, TR>(rhs); // Construct result
    bh_matmul (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}



template <typename T>
inline multi_array<bool>& eq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_equal (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& eq (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& eq (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator== (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_equal (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator== (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator== (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& neq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_not_equal (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& neq (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_not_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& neq (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_not_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator!= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_not_equal (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator!= (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_not_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator!= (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_not_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& gt (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_greater (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& gt (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_greater (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& gt (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_greater (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator> (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_greater (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator> (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_greater (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator> (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_greater (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& gteq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_greater_equal (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& gteq (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_greater_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& gteq (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_greater_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator>= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_greater_equal (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator>= (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_greater_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator>= (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_greater_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& lt (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_less (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& lt (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_less (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& lt (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_less (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator< (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_less (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator< (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_less (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator< (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_less (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& lteq (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_less_equal (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& lteq (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_less_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& lteq (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_less_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator<= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }
    
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(*left); // Construct result
    bh_less_equal (*res, *left, *right); // Encode and enqueue
    res->setTemp(true); // Mark res as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator<= (multi_array<T>& lhs, const T rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(lhs); // Construct result
    bh_less_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}

template <typename T>
inline multi_array<bool>& operator<= (const T lhs, multi_array<T>& rhs)
{
    multi_array<bool>* res = &Runtime::instance().create_base<bool, T>(rhs); // Construct result
    bh_less_equal (*res, lhs, rhs); // Encode and enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}


//
//  multi_array - Internally defined operator overloads
//

template <typename T>
inline
multi_array<T>& multi_array<T>::operator+= (const T& rhs)
{
    bh_add (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator+= (multi_array<T>& rhs)
{
    bh_add (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator-= (const T& rhs)
{
    bh_subtract (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator-= (multi_array<T>& rhs)
{
    bh_subtract (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator*= (const T& rhs)
{
    bh_multiply (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator*= (multi_array<T>& rhs)
{
    bh_multiply (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator/= (const T& rhs)
{
    bh_divide (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator/= (multi_array<T>& rhs)
{
    bh_divide (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator%= (const T& rhs)
{
    bh_mod (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator%= (multi_array<T>& rhs)
{
    bh_mod (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator&= (const T& rhs)
{
    bh_bitwise_and (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator&= (multi_array<T>& rhs)
{
    bh_bitwise_and (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator|= (const T& rhs)
{
    bh_bitwise_or (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator|= (multi_array<T>& rhs)
{
    bh_bitwise_or (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator^= (const T& rhs)
{
    bh_bitwise_xor (*this, *this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::operator^= (multi_array<T>& rhs)
{
    bh_bitwise_xor (*this, *this, rhs);
    return *this;
}
}
#endif
