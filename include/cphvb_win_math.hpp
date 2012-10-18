/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
 
#ifndef __CPHVB_WIN_MATH_H
#define __CPHVB_WIN_MATH_H

#ifdef _WIN32

#include <math.h>
#include <cphvb_type.h>

//See: http://stackoverflow.com/questions/758001/log2-not-found-in-my-math-h
inline double log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2.0 );  
}

//http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> inline int sgn(T val) {
    return (val > T(0)) - (val < T(0));
}

//See: http://www.johndcook.com/cpp_expm1.html
//Rewritten to be template based, should be in VS11
template <typename T> inline T expm1(T x)
{
	if (abs(x) < T(1e-5))
		return x + T(0.5)*x*x;
	else
		return exp(x) - T(1.0);
}

//See: http://stackoverflow.com/questions/570669/checking-if-a-double-or-float-is-nan-in-c,
// http://stackoverflow.com/a/570694
template <typename T> inline T isNaN(T x) { return x != x; }

//Yikes, should use a library function!
template <typename T> inline T trunc(T x) { 
	if (isNaN(x))
		return x;
	else {
		T sign = sgn(x);
		x = abs(x);
		if (x < T(1))
			return sign * 0;
		else
			return sgn(x) * floor(x);
	}
}

//Should handle inf and NaN
template <typename T> inline T log1p(T x) {
	if (isNaN(x))
		return x;
	return log(T(1) + x); 
}

//Add some missing overloads, Visual C++ compiler detects these as ambiguous
inline cphvb_uint32 floor( cphvb_uint32 n ) { return n; }
inline cphvb_uint64 floor( cphvb_uint64 n ) { return n; }
inline cphvb_int32 floor( cphvb_int32 n ) { return n; }
inline cphvb_int64 floor( cphvb_int64 n ) { return n; }
inline cphvb_int32 atan( cphvb_int32 n ) { return (cphvb_int32)atan((float)n); }

inline cphvb_int32 pow( cphvb_int32 n, cphvb_int32 e ) { return (cphvb_int32)pow((float)n, e); }
inline cphvb_int64 pow( cphvb_int64 n, cphvb_int64 e ) { return (cphvb_int64)pow((double)n, (double)e); }
inline cphvb_uint64 pow( cphvb_uint64 n, cphvb_uint64 e ) { return (cphvb_uint64)pow((double)n, (double)e); }
inline cphvb_uint32 pow( cphvb_uint32 n, cphvb_uint32 e ) { return (cphvb_uint32)pow((float)n, (float)e); }
inline cphvb_float32 pow( cphvb_int32 n, cphvb_float32 e ) { return (cphvb_float32)pow((float)n, (float)e); }
inline cphvb_float64 pow( cphvb_int32 n, cphvb_float64 e ) { return (cphvb_float64)pow((double)n, (double)e); }

//All the stuff that Microsoft decided not to implement:
// http://msdn.microsoft.com/en-us/library/w3t84e33(v=vs.100).aspx
#define _MAKE_MATH_FUNCS(NAME, CODE) \
	inline double NAME(double n) { return CODE; }\
	inline float NAME(float n) { return CODE; }


_MAKE_MATH_FUNCS(sec, (1 / cos(n)))
_MAKE_MATH_FUNCS(csc, (1 / sin(n)))
_MAKE_MATH_FUNCS(ctan, (1 / tan(n)))
_MAKE_MATH_FUNCS(asec, (2 * atan(1) - atan(sgn(n) / sqrt((n) * (n) - 1))))
_MAKE_MATH_FUNCS(acsc, (atan(sgn(n) / sqrt((n) * (n) - 1))))
_MAKE_MATH_FUNCS(acot, (2 * atan(1) - atan(n)))
_MAKE_MATH_FUNCS(sech, (2 / (exp(n) + exp(-(n)))))
_MAKE_MATH_FUNCS(csch, (2 / (exp(n) - exp(-(n)))))
_MAKE_MATH_FUNCS(coth, ((exp(n) + exp(-(n))) / (exp(n) - exp(-(n)))))
_MAKE_MATH_FUNCS(asinh, (log((n) + sqrt((n) * (n) + 1))))
_MAKE_MATH_FUNCS(acosh, (log((n) + sqrt((n) * (n) - 1))))
_MAKE_MATH_FUNCS(atanh, (log((1 + (n)) / (1 - (n))) / 2))
_MAKE_MATH_FUNCS(asech, (log((sqrt(-(n) * (n) + 1) + 1) / (n))))
_MAKE_MATH_FUNCS(acsch, (log((sgn(n) * sqrt((n) * (n) + 1) + 1) / (n))))
_MAKE_MATH_FUNCS(acoth, (log(((n) + 1) / ((n) - 1)) / 2))

#undef _MAKE_MATH_FUNCS

#endif

#endif