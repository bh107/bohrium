#region Copyright
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
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Double;
using NumCIL;
using System.Linq.Expressions;


namespace Tester
{
    public static class Program
    {
        static void Main(string[] args)
        {
            //NumCIL.cphVB.Utility.SetupDebugEnvironmentVariables();

            Console.WriteLine("Tester execution with {0} workblock{1}", NumCIL.UFunc.Threads.BlockCount, NumCIL.UFunc.Threads.BlockCount == 1 ? "" : "s");
            Console.WriteLine("Tester UnsafeAPI is {0}", !NumCIL.UnsafeAPI.DisableUnsafeAPI && NumCIL.UnsafeAPI.IsUnsafeSupported ? "ENABLED" : "DISABLED");
            Console.WriteLine("Tester Unsafe arrays is {0}", !NumCIL.UnsafeAPI.DisableUnsafeAPI && NumCIL.UnsafeAPI.IsUnsafeSupported && !NumCIL.UnsafeAPI.DisableUnsafeArrays ? "ENABLED": "DISABLED");
            Console.WriteLine("Tester Unsafe arrays limit is {0}MB", !NumCIL.UnsafeAPI.DisableUnsafeAPI && NumCIL.UnsafeAPI.IsUnsafeSupported && !NumCIL.UnsafeAPI.DisableUnsafeArrays ? (NumCIL.UnsafeAPI.UnsafeArraysLargerThan/ (1024*1024)) : 0.0);

            try
            {
                TimeJacobi();
                Console.WriteLine("Seconds consumed by threads: {0}", TimeSpan.FromTicks(NumCIL.ThreadPool.TicksExecuted).TotalSeconds);
                NumCIL.ThreadPool.TicksExecuted = 0;
                TimeJacobiFixed();
                Console.WriteLine("Seconds consumed by threads: {0}", TimeSpan.FromTicks(NumCIL.ThreadPool.TicksExecuted).TotalSeconds);
                NumCIL.ThreadPool.TicksExecuted = 0;
                TimeScholes();
                Console.WriteLine("Seconds consumed by threads: {0}", TimeSpan.FromTicks(NumCIL.ThreadPool.TicksExecuted).TotalSeconds);
                NumCIL.ThreadPool.TicksExecuted = 0;
                TimeShallowWater();
                Console.WriteLine("Seconds consumed by threads: {0}", TimeSpan.FromTicks(NumCIL.ThreadPool.TicksExecuted).TotalSeconds);
                NumCIL.ThreadPool.TicksExecuted = 0;
                TimekNN();
                Console.WriteLine("Seconds consumed by threads: {0}", TimeSpan.FromTicks(NumCIL.ThreadPool.TicksExecuted).TotalSeconds);
                NumCIL.ThreadPool.TicksExecuted = 0;
                TimenBody();
                Console.WriteLine("Seconds consumed by threads: {0}", TimeSpan.FromTicks(NumCIL.ThreadPool.TicksExecuted).TotalSeconds);
                NumCIL.ThreadPool.TicksExecuted = 0;

                //NumCIL.cphVB.Utility.Activate();

                //TimeJacobi();
                //TimeJacobiFixed();
                //TimeScholes();
                //TimeShallowWater();
                //TimekNN();
                //TimenBody();
                return;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                //Console.ReadLine();
            }
        }

        private static void TimeJacobi()
        {
            long size = 10000;
            double delta;
            using (new DispTimer(string.Format("JacobiSolver {0}x{0}", size)))
                delta = JacobiSolver.Solve(size, size, true, 10);

            Console.WriteLine("Delta: " + delta.ToString());
        }

        private static void TimeJacobiFixed()
        {
            long size = 10000;
            double chk;
            using (new DispTimer(string.Format("JacobiSolverFixed {0}x{0}", size)))
                chk = JacobiSolver.Solve(size, size, false, 10);

            Console.WriteLine("Check: " + chk.ToString());
        }

        private static void TimeScholes()
        {
            long size = 3200000;
            long years = 10;
            double result;

            using (new DispTimer(string.Format("BlackSholes {0}x{1}", size, years)))
                result = BlackScholesSolver.Solve(size, years);

            Console.WriteLine("Result: " + result.ToString());
        }

        private static void TimekNN()
        {
            long size = 10000;
            long dims = 120;
            long k = 4;
            NdArray result;

            using (new DispTimer(string.Format("kNN {0}x{1}, k={2}", size, dims, k)))
                result = kNNSolver.Solve(size, dims, k);

            Console.WriteLine("Result: " + result.ToString());
        }

		private static void TimenBody()
        {
            long size = 5000;
            long steps = 10;

			using (new DispTimer(string.Format("nBody {0}x{1}", size, steps)))
                nBodySolver.Solve(size, steps);
        }

        private static void TimeShallowWater()
        {
            long size = 5000;
            long timesteps = 10;
            double r;
            using (new DispTimer(string.Format("ShallowWaterSolver {0}x{0} with {1} rounds", size, timesteps)))
                r = ShallowWaterSolver.Solve(size, timesteps);

            Console.WriteLine("Result: {0}", r);
        }



    }
}
