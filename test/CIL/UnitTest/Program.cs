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

namespace UnitTest
{
    class Program
    {
        static void Main(string[] args)
		{
			NumCIL.Bohrium.Utility.SetupDebugEnvironmentVariables();

			if (!args.Contains<string>("--bohrium-only", StringComparer.InvariantCultureIgnoreCase))
			{
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
			}
			
            try { NumCIL.Bohrium2.Utility.Activate(); }
            catch (Exception ex) 
            {
                Console.WriteLine(ex.ToString());
            } 

            if (NumCIL.Generic.NdArray<float>.AccessorFactory.GetType() == typeof(NumCIL.Bohrium2.BohriumAccessorFactory<float>))
                RunSomeTests("Bohrium");
            else
                Console.WriteLine("Bohrium code is not supported, skipping tests for Bohrium code");

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

            Console.WriteLine("Running repeat tests" + name);
            using (new DispTimer("Repeat tests"))
                RepeatTests.RunTests();

            Console.WriteLine("Running complex tests" + name);
            using (new DispTimer("Complex tests"))
                ComplexTests.RunTests();

            Console.WriteLine("Running logical tests" + name);
            using (new DispTimer("Logical tests"))
                LogicalTests.RunTests();

			Console.WriteLine("Running benchmark tests" + name);
            using (new DispTimer("benchmark tests"))
                BenchmarkTests.RunTests();
        }
    }
}
