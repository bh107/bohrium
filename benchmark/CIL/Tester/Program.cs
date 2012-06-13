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
            NumCIL.cphVB.Utility.SetupDebugEnvironmentVariables();

            int threads;
            if (int.TryParse(Environment.GetEnvironmentVariable("NUMCIL_THREADS"), out threads))
            {
                int p1, p2;
                System.Threading.ThreadPool.GetMaxThreads(out p1, out p2);
                System.Threading.ThreadPool.SetMaxThreads(threads, p2);
            }

            Console.WriteLine("Tester execution with {0} workblock{1}", NumCIL.UFunc.Threads.BlockCount, NumCIL.UFunc.Threads.BlockCount == 1 ? "" : "s");
            Console.WriteLine("Tester UnsafeAPI is {0}", !NumCIL.UnsafeAPI.DisableUnsafeAPI && NumCIL.UnsafeAPI.IsUnsafeSupported ? "ENABLED" : "DISABLED");
            Console.WriteLine("Tester Unsafe arrays is {0}", !NumCIL.UnsafeAPI.DisableUnsafeAPI && NumCIL.UnsafeAPI.IsUnsafeSupported && !NumCIL.UnsafeAPI.DisableUnsafeArrays ? "ENABLED": "DISABLED");
            Console.WriteLine("Tester Unsafe arrays limit is {0}MB", !NumCIL.UnsafeAPI.DisableUnsafeAPI && NumCIL.UnsafeAPI.IsUnsafeSupported && !NumCIL.UnsafeAPI.DisableUnsafeArrays ? (NumCIL.UnsafeAPI.UnsafeArraysLargerThan/ (1024*1024)) : 0.0);

            try
            {
                TimeJacobi();
                TimeScholes();
                TimeJacobiFixed();
                TimeShallowWater();
                TimekNN();

                //NumCIL.cphVB.Utility.Activate();

                //TimeJacobi();
                //TimeScholes();
                //TimeJacobiFixed();
                //TimeShallowWater();
                //TimekNN();
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
            long count;
            using (new DispTimer(string.Format("JacobiSolver {0}x{0}", size)))
                count = JacobiSolver.Solve(size, size);

            Console.WriteLine("Count: " + count.ToString());
        }

        private static void TimeJacobiFixed()
        {
            long size = 10000;
            long count;
            using (new DispTimer(string.Format("JacobiSolverFixed {0}x{0}", size)))
                count = JacobiSolver.Solve(size, size, 2000);

            Console.WriteLine("Count: " + count.ToString());
        }

        private static void TimeScholes()
        {
            long size = 3200000;
            long years = 36;
            double result;

            using (new DispTimer(string.Format("BlackSholes {0}x{1}", size, years)))
                result = BlackScholesSolver.Solve(size, years);

            Console.WriteLine("Result: " + result.ToString());
        }

        private static void TimekNN()
        {
            long size = 2000;
            long dims = 64;
            long k = 4;
            NdArray result;

            using (new DispTimer(string.Format("kNN {0}x{1}, k={2}", size, dims, k)))
                result = kNNSolver.Solve(size, dims, k);

            Console.WriteLine("Result: " + result.ToString());
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
