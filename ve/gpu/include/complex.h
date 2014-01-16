/* inspired by pyopencl-complex.h */
/* TODO: implement implement add, sub mul... with real.
 */
#define logmaxfloat  log(FLT_MAX)
#define logmaxdouble log(DBL_MAX)
#define CADD(r,a,b) r = a + b;
#define CSUB(r,a,b) r = a - b;
#define CMUL(r,a,b) r.s0 = a.s0*b.s0 - a.s1*b.s1;               \
                    r.s1 = a.s0*b.s1 + a.s1*b.s0;
#define CDIV(t,r,x,y) { t ratio, denom, a, b, c, d;             \
                      if (fabs(y.s0) <= fabs(y.s1)) {           \
                          ratio = y.s0 / y.s1;                  \
                          denom = y.s1;                         \
                          a = x.s1;                             \
                          b = x.s0;                             \
                          c = -x.s0;                            \
                          d = x.s1;                             \
                      } else {                                  \
                          ratio = y.s1 / y.s0;                  \
                          denom = y.s0;                         \
                          a = x.s0;                             \
                          b = x.s1;                             \
                          c = x.s1;                             \
                          d = -x.s0;                            \
                      }                                         \
                      denom *= (1 + ratio * ratio);             \
                      r.s0 = (a + b * ratio) / denom;           \
                      r.s1 = (c + d * ratio) / denom; }
#define CEQ(r,a,b) r = (a.s0 == b.s0) && (a.s1 == b.s1);
#define CNEQ(r,a,b) r = (a.s0 != b.s0) || (a.s1 != b.s1);
#define CPOW(t,r,a,b) { t logr = log(hypot(a.s0, a.s1));        \
                        t logi = atan2(a.s1, a.s0);             \
                        t x = exp(logr * b.s0 - logi * b.s1);   \
                        t y = logr * b.s1 + logi * b.s0;        \
                        r.s1 = sincos(y, &r.s0);                \
                        r = x*r; }
#define CSQRT(r,a) r.s0 = hypot(a.s0, a.s1);                    \
                   if (r.s0 == 0.0)                             \
                       r.s0 = r.s1 = 0.0;                       \
                   else if (a.s0 > 0.0) {                       \
                       r.s0 = sqrt(0.5 * (r.s0 + a.s0));        \
                       r.s1 = a.s1/r.s0/2.0;                    \
                   } else {                                     \
                       r.s1 = sqrt(0.5 * (r.s0 - a.s0));        \
                       if (a.s1 < 0.0)                          \
                           r.s1 = -r.s1;                        \
                       r.s0 = a.s1/r.s1/2.0;                    \
                   }
#define CEXP(t,r,a) { t cosi, sini, expr;                       \
                      expr = exp(a.s0);                         \
                      sini = sincos(a.s1, &cosi);               \
                      r.s0 = expr*cosi;                         \
                      r.s1 = expr*sini; }
#define CLOG(r,a) r.s0 = log(hypot(a.s0, a.s1));                \
                  r.s1 = atan2(a.s1, a.s0);
#define CSIN(t,r,a) { t cosr, sinr;                             \
                    sinr = sincos(a.s0, &cosr);                 \
                    r.s0 = sinr*cosh(a.s1);                     \
                    r.s1 = cosr*sinh(a.s1); }
#define CCOS(t, r,a) { t cosr, sinr;                            \
                     sinr = sincos(a.s0, &cosr);                \
                     r.s0 = cosr*cosh(a.s1);                    \
                     r.s1 = -sinr*sinh(a.s1); }
#define CTAN(t,r,a) r = 2.0*a;                                  \
                    if (fabs(r.s1) > logmax##t) {               \
                        r.s0 = 0.0;                             \
                        r.s1 = (r.s1 > 0.0 ? 1.0 : -1.0);       \
                    } else {                                    \
                        t d = cos(r.s0) + cosh(r.s1);           \
                        r.s0 = sin(r.s0) / d;                   \
                        r.s1 = sinh(r.s1) / d;                  \
                    }
#define CSINH(t,r,a) { t cosi, sini;                            \
                     sini = sincos(a.s1, &cosi);                \
                     r.s0 = sinh(a.s0)*cosi;                    \
                     r.s1 = cosh(a.s0)*sini; }
#define CCOSH(t,r,a) { t cosi, sini;                            \
                     sini = sincos(a.s1, &cosi);                \
                     r.s0 = cosh(a.s0)*cosi;                    \
                     r.s1 = sinh(a.s0)*sini; }
#define CTANH(t,r,a) r = 2.0*a;                                 \
                     if (fabs(r.s0) > logmax##t) {              \
                         r.s0 = (r.s0 > 0.0 ? 1.0 : -1.0);      \
                         r.s1 = 0.0;                            \
                     } else {                                   \
                         t d = cosh(r.s0) + cos(r.s1);          \
                         r.s0 = sinh(r.s0) / d;                 \
                         r.s1 = sin(r.s1) / d;                  \
                     }
