using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Float;
using NumCIL;
using System.Linq.Expressions;


namespace Tester
{
    public static class Program
    {
        static void Main(string[] args)
        {
			//Bad OS detection :)
			if (System.IO.Path.PathSeparator == ':')
			{
	            Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", @"/Users/kenneth/Udvikler/cphvb/core:/Users/kenneth/Udvikler/cphvb/vem/node");
	            Environment.SetEnvironmentVariable("DYLD_LIBRARY_PATH", @"/Users/kenneth/Udvikler/cphvb/core:/Users/kenneth/Udvikler/cphvb/vem/node");
	            Environment.SetEnvironmentVariable("CPHVB_CONFIG", @"/Users/kenneth/Udvikler/cphvb/config.osx.ini");
			}
			else
			{
				string path = Environment.GetEnvironmentVariable("PATH");
	            Environment.SetEnvironmentVariable("PATH", path + @";Z:\Udvikler\cphvb\core;Z:\Udvikler\cphvb\vem\node;Z:\Udvikler\cphvb\pthread_win32");
	            Environment.SetEnvironmentVariable("CPHVB_CONFIG", @"Z:\Udvikler\cphvb\config.win.ini");
                //Environment.SetEnvironmentVariable("CPHVB_CONFIG", @"config.ini");
                //Environment.SetEnvironmentVariable("PATH", path + @";" + System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location));
			}


            try
            {
                //TimeJacobi();
                //TimeScholes();
                //TimeJacobiFixed();
                TimeShallowWater();
                //TimekNN();

                NumCIL.cphVB.Utility.Activate();

                //TimeJacobi();
                //TimeScholes();
                //TimeJacobiFixed();
                TimeShallowWater();
                //TimekNN();
                return;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.ReadLine();
            }
        }

        private static void TimeJacobi()
        {
            long size = 100;
            long count;
            using (new DispTimer(string.Format("JacobiSolver {0}x{0}", size)))
                count = JacobiSolver.Solve(size, size);

            Console.WriteLine("Count: " + count.ToString());
        }

        private static void TimeJacobiFixed()
        {
            long size = 5000;
            long count;
            using (new DispTimer(string.Format("JacobiSolverFixed {0}x{0}", size)))
                count = JacobiSolver.Solve(size, size, 10);

            Console.WriteLine("Count: " + count.ToString());
        }

        private static void TimeScholes()
        {
            long size = 320000;
            long years = 36;
            double result;

            using (new DispTimer(string.Format("BlackSholes {0}x{1}", size, years)))
                result = BlackScholesSolver.Solve(size, years);

            Console.WriteLine("Result: " + result.ToString());
        }

        private static void TimekNN()
        {
            long size = 1000;
            long dims = 64;
            long k = 4;
            NdArray result;

            using (new DispTimer(string.Format("kNN {0}x{1}, k={2}", size, dims, k)))
                result = kNNSolver.Solve(size, dims, k);

            Console.WriteLine("Result: " + result.ToString());


        }

        private static void TimeShallowWater()
        {
            long size = 1000;
            long timesteps = 10;
            float r;
            using (new DispTimer(string.Format("ShallowWaterSolver {0}x{0} with {1} rounds", size, timesteps)))
                r = ShallowWaterSolver.Solve(size, timesteps);

            Console.WriteLine("Result: {0}", r);
        }



    }
}
