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
            var jacobiResult = Tester.JacobiSolver.Solve(100, 100);
            if (jacobiResult != 8349)
                throw new Exception(string.Format("Jacobi solver failed: {0}", jacobiResult));

            var blackScholesResult = Tester.BlackScholesSolver.Solve(36000, 10, false);
            if (Math.Abs(blackScholesResult - 51.855494520660294) > 0.01)
                throw new Exception(string.Format("BlackScholes solver failed: {0}, diff: {1}", blackScholesResult, Math.Abs(blackScholesResult - 51.855494520660294)));

            var kNNResult = Tester.kNNSolver.Solve(100, 64, 4, false).Sum();
            if (Math.Abs(kNNResult - 9969336320) > 0.01)
                throw new Exception(string.Format("kNN solver failed: {0}, diff: {1}", kNNResult, Math.Abs(kNNResult - 9969336320)));

            var shallowWaterResult = Tester.ShallowWaterSolver.Solve(200, 4);
            if (Math.Abs(shallowWaterResult - 204.04741883859941) > 0.01)
                throw new Exception(string.Format("ShallowWater solver failed: {0}, diff {1}", shallowWaterResult, Math.Abs(shallowWaterResult - 204.04741883859941)));

        }
    }
}
