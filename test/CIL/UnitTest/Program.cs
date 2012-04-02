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
            Console.WriteLine("Running profiling tests");
            using (new DispTimer("Profiling tests"))
            {
                Profiling.RunProfiling();
            }
        }
    }
}
