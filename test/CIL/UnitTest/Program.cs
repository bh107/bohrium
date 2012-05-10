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
            NumCIL.UnsafeAPI.DisableUnsafeAPI = true;

            Console.WriteLine("Running basic tests");
            using (new DispTimer("Basic tests"))
                BasicTests.RunTests();

            Console.WriteLine("Running Lookup tests");
            using (new DispTimer("Lookup tests"))
                TypeLookupTests.RunTests();

            Console.WriteLine("Running extended tests");
            using (new DispTimer("Extended tests"))
                ExtendedTests.RunTests();

            if (NumCIL.UnsafeAPI.IsUnsafeSupported)
            {
                NumCIL.UnsafeAPI.DisableUnsafeAPI = false;

                Console.WriteLine("Running basic tests - Unsafe");
                using (new DispTimer("Basic tests"))
                    BasicTests.RunTests();

                Console.WriteLine("Running Lookup tests - Unsafe");
                using (new DispTimer("Lookup tests"))
                    TypeLookupTests.RunTests();

                Console.WriteLine("Running extended tests - Unsafe");
                using (new DispTimer("Extended tests"))
                    ExtendedTests.RunTests();
            }
            else
            {
                Console.WriteLine("Unsafe code is not supported, skipping tests for unsafe code");
            }

            NumCIL.Generic.NdArray<float>.AccessorFactory = new NumCIL.Generic.LazyAccessorFactory<float>();

            Console.WriteLine("Running basic tests - Lazy");
            using (new DispTimer("Basic tests"))
                BasicTests.RunTests();

            Console.WriteLine("Running Lookup tests - Lazy");
            using (new DispTimer("Lookup tests"))
                TypeLookupTests.RunTests();

            Console.WriteLine("Running extended tests");
            using (new DispTimer("Extended tests"))
                ExtendedTests.RunTests();

            /*Console.WriteLine("Running profiling tests");
            using (new DispTimer("Profiling tests"))
                Profiling.RunProfiling();*/
        }
    }
}
