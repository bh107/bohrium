using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Double;

namespace Tester
{
    using DATA = System.Double;

    public static class BlackScholesSolver
    {
        private struct LessThan : NumCIL.IUnaryOp<DATA> { public DATA Op(DATA a) { return a < 0 ? 1 : 0; } }
        private struct GreaterThanOrEqual : NumCIL.IUnaryOp<DATA> { public DATA Op(DATA a) { return a >= 0 ? 1 : 0; } }

        private static NdArray CND(NdArray X)
        {
            DATA a1 = 0.31938153f, a2 = -0.356563782f, a3 = 1.781477937f, a4 = -1.821255978f, a5 = 1.330274429f;
            var L = X.Abs();
            var K = 1.0f / (1.0f + 0.2316419f * L);
            var w = 1.0f - 1.0f / ((DATA)Math.Sqrt(2 * Math.PI)) * (L.Negate() * L / 2.0f).Exp() * (a1 * K + a2 * (K.Pow(2)) + a3 * (K.Pow(3)) + a4 * (K.Pow(4)) + a5 * (K.Pow(5)));
            
            var mask1 = X.Apply<LessThan>();
            var mask2 = X.Apply<GreaterThanOrEqual>();

            w = w * mask2 + (1.0f - w) * mask1;
            return w;
        }

        private static NdArray BlackSholes(bool callputflag, NdArray S, DATA X, DATA T, DATA r, DATA v)
        {
            var d1 = ((S / X).Log() + (r + v * v / 2.0f) * T) / (v * (DATA)Math.Sqrt(T));
            var d2 = d1 - v * (DATA)Math.Sqrt(T);

            if (callputflag)
                return S * CND(d1) - X * (DATA)Math.Exp(-r * T) * CND(d2);
            else
                return X * (DATA)Math.Exp(-r * T) * CND(d2.Negate()) - S * CND(d1.Negate());
        }

        public static DATA Solve(long size, long years, bool randomdata = true)
        {
            var S = randomdata ? Generate.Random(size) : Generate.Ones(size);
            S = S * 4.0f - 2.0f + 60.0f; //Price is 58-62

            DATA X = 65.0f;
            DATA r = 0.08f;
            DATA v = 0.3f;

            DATA day=1.0f/years;
            DATA T = day;

            DATA total = 0.0f;

            for (long i = 0; i < years; i++)
            {
                total += Add.Reduce(BlackSholes(true, S, X, T, r, v)).Value[0] / size;
                T += day;
            }

            return total;
        }
    }
}
