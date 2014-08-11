#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium:
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
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Double;

namespace UnitTest.Benchmarks
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
            var src = randomdata ? Generate.Random(size, dimensions) : Generate.Range(size, dimensions);
            return kNN(src)[R.Slice(0, k)];
        }
    }
}
