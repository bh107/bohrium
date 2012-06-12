using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Double;

namespace Tester
{
    using T = System.Double;
    using R = NumCIL.Range;

    public static class kNNSolver
    {
        private static NdArray ComputeTargets(NdArray src, NdArray targets)
        {
            var b1 = src[R.All, R.NewAxis, R.All];
            var d1 = (b1 - targets).Pow(2);
            var d2 = Add.Reduce(d1, 2);
            var d3 = d2.Sqrt();
            var r = Max.Reduce(d2, 1);
            return r;
        }

        private static NdArray kNN(NdArray input)
        {
            var src = input.Clone();
            var targets = input.Clone();

            return ComputeTargets(src, input);
        }

        public static NdArray Solve(long size, long dimensions, long k, bool randomdata = true)
        {
            var src = randomdata ? Generate.Random(size, dimensions) : Generate.Arange(size, dimensions);
            return kNN(src)[R.Slice(0, k)];
        }
    }
}
