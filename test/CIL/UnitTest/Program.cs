using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace UnitTest
{
    class Program
    {
        static void Main(string[] args)
        {
            NumCIL.cphVB.Utility.SetupDebugEnvironmentVariables();

            NumCIL.UnsafeAPI.DisableUnsafeAPI = true;
            RunSomeTests(null);

            if (NumCIL.UnsafeAPI.IsUnsafeSupported)
            {
                NumCIL.UnsafeAPI.DisableUnsafeAPI = false;
                RunSomeTests("Unsafe");
            }
            else
                Console.WriteLine("Unsafe code is not supported, skipping tests for unsafe code");

            NumCIL.Generic.NdArray<float>.AccessorFactory = new NumCIL.Generic.LazyAccessorFactory<float>();
            RunSomeTests("Lazy");

            try { NumCIL.cphVB.Utility.Activate(); }
            catch { } 

            if (NumCIL.Generic.NdArray<float>.AccessorFactory.GetType() == typeof(NumCIL.cphVB.cphVBAccessorFactory<float>))
                RunSomeTests("cphVB");
            else
                Console.WriteLine("cphVB code is not supported, skipping tests for cphVB code");

            if (args.Contains<string>("--profiling", StringComparer.InvariantCultureIgnoreCase))
            {
                Console.WriteLine("Running profiling tests");
                using (new DispTimer("Profiling tests"))
                    Profiling.RunProfiling();
            }
        }

        private static void RunSomeTests(string name)
        {
            if (!string.IsNullOrEmpty(name))
                name = " - " + name;

            Console.WriteLine("Running basic tests" + name);
            using (new DispTimer("Basic tests"))
                BasicTests.RunTests();

            Console.WriteLine("Running Lookup tests" + name);
            using (new DispTimer("Lookup tests"))
                TypeLookupTests.RunTests();

            Console.WriteLine("Running extended tests" + name);
            using (new DispTimer("Extended tests"))
                ExtendedTests.RunTests();

            Console.WriteLine("Running benchmark tests" + name);
            using (new DispTimer("benchmark tests"))
                BenchmarkTests.RunTests();
        }
    }
}
