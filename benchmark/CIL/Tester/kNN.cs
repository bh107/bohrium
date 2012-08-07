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
            var @base = src[R.All, R.NewAxis];
            var target = src[R.All, R.All, R.NewAxis];

            var tmp = (@base - target).Pow(2);
            tmp = Add.Reduce(tmp);
            Sqrt.Apply(tmp, tmp);

            return Max.Reduce(tmp);
        }

        private static NdArray kNN(NdArray input)
        {
            return ComputeTargets(input, input.Clone());
        }

        public static NdArray Solve(long size, long dimensions, long k, bool randomdata = true)
        {
            var src = randomdata ? Generate.Random(size, dimensions) : Generate.Arange(size, dimensions);
            return kNN(src)[R.Slice(0, k)];
        }
    }
}
