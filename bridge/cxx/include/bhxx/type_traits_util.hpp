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

#pragma once

#include <type_traits>
#include <complex>

namespace bhxx {
namespace type_traits {

/** Type trait to implement `is_safe_numeric_cast`
 *  Based on <https://stackoverflow.com/a/36272533>
 */

template<class T>
struct is_complex_or_floating_point : std::is_floating_point<T> {
};

template<class T>
struct is_complex_or_floating_point<std::complex<T> > : std::is_floating_point<T> {
};

template<class T>
struct is_arithmetic : std::integral_constant<bool,
        std::is_integral<T>::value ||
        is_complex_or_floating_point<T>::value> {
};

// pred_base selects the appropriate base type (true_type or false_type) to
// make defining our own predicates easier.

template<bool>
struct pred_base : std::false_type {
};
template<>
struct pred_base<true> : std::true_type {
};

// same_decayed
// -------------
// Are the decayed versions of "T" and "O" the same basic type?
// Gets around the fact that std::is_same will treat, say "bool" and "bool&" as
// different types and using std::decay all over the place gets really verbose

template<class T, class O>
struct same_decayed : pred_base<std::is_same<typename std::decay<T>::type, typename std::decay<O>::type>::value> {
};


// is_numeric.  Is it a number?  i.e. true for floats and integrals but not bool

template<class T>
struct is_numeric : pred_base<is_arithmetic<T>::value && !same_decayed<bool, T>::value> {
};


// both - less verbose way to determine if TWO types both meet a single predicate

template<class A, class B, template<typename> class PRED>
struct both : pred_base<PRED<A>::value && PRED<B>::value> {
};

// Some simple typedefs of both (above) for common conditions

template<class A, class B>
struct both_numeric : both<A, B, is_numeric> {
};    // Are both A and B numeric        types?
template<class A, class B>
struct both_floating : both<A, B, is_complex_or_floating_point> {
};    // Are both A and B floating point types?
template<class A, class B>
struct both_integral : both<A, B, std::is_integral> {
};    // Are both A and B integral       types
template<class A, class B>
struct both_signed : both<A, B, std::is_signed> {
};    // Are both A and B signed         types
template<class A, class B>
struct both_unsigned : both<A, B, std::is_unsigned> {
};    // Are both A and B unsigned       types


// Returns true if both number types are signed or both are unsigned
template<class T, class F>
struct same_signage : pred_base<(both_signed<T, F>::value) || (both_unsigned<T, F>::value)> {
};

// And here, finally is the trait I wanted in the first place:  is_safe_numeric_cast

template<class T, class F>
struct is_safe_numeric_cast
        : pred_base<both_numeric<T, F>::value &&// Obviously both src and dest must be numbers
                    (is_complex_or_floating_point<T>::value &&
                     (std::is_integral<F>::value || sizeof(T) >= sizeof(F))) ||
                    // Floating dest: src must be integral or smaller/equal float-type
                    ((both_integral<T, F>::value) &&
                     // Integral dest: src must be integral and (smaller/equal+same signage) or (smaller+different signage)
                     (sizeof(T) > sizeof(F) || (sizeof(T) == sizeof(F) && same_signage<T, F>::value)))> {
};

// Instantiate all possible types of `BhArray`. Define the macro function INSTANTIATE(TYPE),
// which must define a function prototype of type `TYPE`.
#define instantiate_dtype() INSTANTIATE(bool) INSTANTIATE(int8_t) INSTANTIATE(int16_t) INSTANTIATE(int32_t) \
                            INSTANTIATE(int64_t) INSTANTIATE(uint8_t) INSTANTIATE(uint16_t) INSTANTIATE(uint32_t) \
                            INSTANTIATE(uint64_t) INSTANTIATE(float) INSTANTIATE(double) \
                            INSTANTIATE(std::complex<float>) INSTANTIATE(std::complex<double>)

// Instantiate all possible types of `BhArray` excluding `bool`. Define the macro function INSTANTIATE(TYPE),
// which must define a function prototype of type `TYPE`.
#define instantiate_dtype_excl_bool() INSTANTIATE(int8_t) INSTANTIATE(int16_t) \
                                      INSTANTIATE(int32_t) INSTANTIATE(int64_t) INSTANTIATE(uint8_t) \
                                      INSTANTIATE(uint16_t) INSTANTIATE(uint32_t) INSTANTIATE(uint64_t) \
                                      INSTANTIATE(float) INSTANTIATE(double) INSTANTIATE(std::complex<float>) \
                                      INSTANTIATE(std::complex<double>)

} // namespace type_traits
} // namespace bhxx


