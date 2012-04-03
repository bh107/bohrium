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
				string p = Environment.GetEnvironmentVariable("PATH");
	            Environment.SetEnvironmentVariable("PATH", p + @";Z:\Udvikler\cphvb\core;Z:\Udvikler\cphvb\vem\node;Z:\Udvikler\cphvb\pthread_win32");
	            Environment.SetEnvironmentVariable("CPHVB_CONFIG", @"Z:\Udvikler\cphvb\config.win.ini");
                //Environment.SetEnvironmentVariable("CPHVB_CONFIG", @"config.ini");
                //Environment.SetEnvironmentVariable("PATH", p + @";" + System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location));
			}


            try
            {
                TimeJacobi();
                //TimeScholes();
                
                NumCIL.cphVB.Utility.Activate();

                TimeJacobi();
                //TimeScholes();
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
            long size = 100;
            long count;
            using (new DispTimer(string.Format("JacobiSolverFixed {0}x{0}", size)))
                count = JacobiSolver.Solve(size, size, 8349);

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

        private static void basicTests()
        {
            var test = Generate.Arange(3) * 4;

            Shape s = new Shape(
                new long[] { 2, 1, 2, 3 },  //Dimension sizes
                6,                          //Offset
                new long[] { 18, 18, 6, 1 } //Strides
            );

            var a = Generate.Arange(s.Length);
            var b = a.Reshape(s);
            var c = b[1][0][1];
            var d = c[2];

            foreach (var v in b[1, 0, 1].Value)
                Console.WriteLine("{0}", v);

            float n = b.Value[1, 0, 1, 2];
            n = c.Value[2];
            n = d.Value[0];
            n = b.Value[1];

            var r1 = Generate.Arange(12).Reshape(new long[] { 2, 1, 2, 3 });
            var r2 = r1.Reduce<Add>(0);
            r2 = r1.Reduce<Add>(1);
            r2 = r1.Reduce<Add>(2);
            r2 = r1.Reduce<Add>(3);

            var r3 = b.Reduce<Add>();

            var x1 = Generate.Arange(12).Reshape(new long[] { 4, 3 });
            var x2 = Generate.Arange(3);

            var x3 = x1 + x2;

            var sqrd = x3.Sqrt();
            sqrd *= sqrd;
            sqrd = sqrd.Round();
            sqrd += 4;
            sqrd++;

            var x5 = sqrd.Apply((x) => x % 2 == 0 ? x : -x);

            NumCIL.UFunc.Apply<float, Add>(x1, x2, x3);
            NumCIL.Double.NdArray x4 = (NumCIL.Double.NdArray)x3;

            var x6 = Generate.Arange(6).Reshape(new long[] { 2, 3 });

            var x7 = x6.Reduce<Add>();


            Console.WriteLine(x7);
        }
    }
}
