#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium:
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
using NumCIL;
using Utilities;

namespace jacobi
{
	public static class Jacobi
    {
		public static void Main (string[] args)
		{
			Utilities.RunBenchmark.Run(args, 2,
               // Running the benchmark
               (input) => {
					var size = input.sizes[0];
					var iterations = input.sizes[1];
					
					if (input.type == typeof(double)) {
						var data = JacobiSolverDouble.Create(size, size);
						using (new DispTimer(string.Format("Jacobi (Double) {0}*{1}", size, iterations)))
							JacobiSolverDouble.Solve(data, false, iterations);
					} else {
						var data = JacobiSolverSingle.Create (size, size);
						using (new DispTimer(string.Format("Jacobi (Float) {0}*{1}", size, iterations)))
							JacobiSolverSingle.Solve(data, false, iterations);
					}
				}
			);
		}
    }
}
