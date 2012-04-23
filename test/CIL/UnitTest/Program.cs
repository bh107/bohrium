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
            Console.WriteLine("Running basic tests");
            using (new DispTimer("Basic tests"))
                BasicTests.RunTests();

            Console.WriteLine("Running Lookup tests");
            using (new DispTimer("Lookup tests"))
                TypeLookupTests.RunTests();

            Console.WriteLine("Running transpose tests");
            using (new DispTimer("Transpose tests"))
                TransposeTests.RunTests();


            NumCIL.Generic.NdArray<float>.AccessorFactory = new NumCIL.Generic.LazyAccessorFactory<float>();

            Console.WriteLine("Running basic tests - Lazy");
            using (new DispTimer("Basic tests"))
                BasicTests.RunTests();

            Console.WriteLine("Running Lookup tests - Lazy");
            using (new DispTimer("Lookup tests"))
                TypeLookupTests.RunTests();

            /*Console.WriteLine("Running profiling tests");
            using (new DispTimer("Profiling tests"))
                Profiling.RunProfiling();*/
        }
    }
}
