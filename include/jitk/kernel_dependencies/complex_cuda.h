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

/* inspired by pyopencl-complex.h */
#pragma once

#include <cuComplex.h>

#define make_complex64(x, y)  make_cuFloatComplex(x, y)
#define make_complex128(x, y) make_cuDoubleComplex(x, y)

// CUDA doesn't define FLT_MAX and DBL_MAX, use values from cuComplex.h
#ifndef FLT_MAX
    #define FLT_MAX (3.402823466e38f)
#endif
#ifndef DBL_MAX
    #define DBL_MAX (1.79769313486231570e+308)
#endif

#define logmaxfloat  log(FLT_MAX)
#define logmaxdouble log(DBL_MAX)

#define CABS(r,a)   r = hypot(a.x, a.y);

#define CADD(r,a,b) r.x = a.x + b.x; \
                    r.y = a.y + b.y;

#define CSUB(r,a,b) r.x = a.x - b.x; \
                    r.y = a.y - b.y;

#define CMUL(r,a,b) r.x = a.x*b.x - a.y*b.y; \
                    r.y = a.x*b.y + a.y*b.x;

#define CDIV(t,r,m,n) { t ratio, denom, a, b, c, d;  \
                      if (fabs(n.x) <= fabs(n.y)) {  \
                          ratio = n.x / n.y;         \
                          denom = n.y;               \
                          a = m.y;                   \
                          b = m.x;                   \
                          c = -m.x;                  \
                          d = m.y;                   \
                      } else {                       \
                          ratio = n.y / n.x;         \
                          denom = n.x;               \
                          a = m.x;                   \
                          b = m.y;                   \
                          c = m.y;                   \
                          d = -m.x;                  \
                      }                              \
                      denom *= (1 + ratio * ratio);  \
                      r.x = (a + b * ratio) / denom; \
                      r.y = (c + d * ratio) / denom; }

#define CEQ(r,a,b) r = (a.x == b.x) && (a.y == b.y);

#define CNEQ(r,a,b) r = (a.x != b.x) || (a.y != b.y);

#define CPOW(t,r,a,b) { t logr, logi, x, y, cosy, siny;   \
                        logr = log(hypot(a.x, a.y));      \
                        logi = atan2(a.y, a.x);           \
                        x = exp(logr * b.x - logi * b.y); \
                        y = logr * b.y + logi * b.x;      \
                        sincos(y, &siny, &cosy);          \
                        r.x = x*cosy;                     \
                        r.y = x*siny; }

#define CSQRT(r,a) r.x = hypot(a.x, a.y);             \
                   if (r.x == 0.0)                    \
                       r.x = r.y = 0.0;               \
                   else if (a.x > 0.0) {              \
                       r.x = sqrt(0.5 * (r.x + a.x)); \
                       r.y = a.y/r.x/2.0;             \
                   } else {                           \
                       r.y = sqrt(0.5 * (r.x - a.x)); \
                       if (a.y < 0.0)                 \
                           r.y = -r.y;                \
                       r.x = a.y/r.y/2.0;             \
                   }

#define CEXP(t,r,a) { t cosi, sini, expr;        \
                      expr = exp(a.x);           \
                      sincos(a.y, &sini, &cosi); \
                      r.x = expr*cosi;           \
                      r.y = expr*sini; }

#define CLOG(r,a) r.x = log(hypot(a.x, a.y)); \
                  r.y = atan2(a.y, a.x);

#define CSIN(t,r,a) { t cosr, sinr;            \
                    sincos(a.x, &sinr, &cosr); \
                    r.x = sinr*cosh(a.y);      \
                    r.y = cosr*sinh(a.y); }

#define CCOS(t, r,a) { t cosr, sinr;            \
                     sincos(a.x, &sinr, &cosr); \
                     r.x = cosr*cosh(a.y);      \
                     r.y = -sinr*sinh(a.y); }

#define CTAN(t,r,a) r.x = 2.0*a.x;                      \
                    r.y = 2.0*a.y;                      \
                    if (fabs(r.y) > logmax##t) {        \
                        r.x = 0.0;                      \
                        r.y = (r.y > 0.0 ? 1.0 : -1.0); \
                    } else {                            \
                        t d = cos(r.x) + cosh(r.y);     \
                        r.x = sin(r.x) / d;             \
                        r.y = sinh(r.y) / d;            \
                    }

#define CSINH(t,r,a) { t cosi, sini;            \
                     sincos(a.y, &sini, &cosi); \
                     r.x = sinh(a.x)*cosi;      \
                     r.y = cosh(a.x)*sini; }

#define CCOSH(t,r,a) { t cosi, sini;            \
                     sincos(a.y, &sini, &cosi); \
                     r.x = cosh(a.x)*cosi;      \
                     r.y = sinh(a.x)*sini; }

#define CTANH(t,r,a) r.x = 2.0*a.x;                      \
                     r.y = 2.0*a.y;                      \
                     if (fabs(r.x) > logmax##t) {        \
                         r.x = (r.x > 0.0 ? 1.0 : -1.0); \
                         r.y = 0.0;                      \
                     } else {                            \
                         t d = cosh(r.x) + cos(r.y);     \
                         r.x = sinh(r.x) / d;            \
                         r.y = sin(r.y) / d;             \
                     }
