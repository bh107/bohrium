#region Copyright
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
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Float;
using NumCIL;
using T = System.Single;

namespace UnitTest
{
    public class BenchmarkTests
    {
        public static void RunTests()
        {
			var jacobiResult = UnitTest.Benchmarks.JacobiSolver.Solve(100, 100, true, 10);
            if (Math.Abs(jacobiResult - 6529.6474649484189) > 0.01)
                throw new Exception(string.Format("Jacobi solver failed: {0}, diff: {1}", jacobiResult, Math.Abs(jacobiResult - 6529.6474649484189)));

			var blackScholesResult = UnitTest.Benchmarks.BlackScholesSolver.Solve(36000, 10, false);
            if (Math.Abs(blackScholesResult - 51.855494520660294) > 0.01)
                throw new Exception(string.Format("BlackScholes solver failed: {0}, diff: {1}", blackScholesResult, Math.Abs(blackScholesResult - 51.855494520660294)));

			var kNNResult = UnitTest.Benchmarks.kNNSolver.Solve(100, 64, 4, false).Sum();
            if (Math.Abs(kNNResult - 2460) > 0.01)
                throw new Exception(string.Format("kNN solver failed: {0}, diff: {1}", kNNResult, Math.Abs(kNNResult - 9969336320)));

			var shallowWaterResult = UnitTest.Benchmarks.ShallowWaterSolver.Solve(200, 4);
            if (Math.Abs(shallowWaterResult - 204.04741883859941) > 0.01)
                throw new Exception(string.Format("ShallowWater solver failed: {0}, diff {1}", shallowWaterResult, Math.Abs(shallowWaterResult - 204.04741883859941)));

			UnitTest.Benchmarks.nBodySolver.Solve(100, 2);
        }
    }
}
