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

//Adapted from: http://people.sc.fsu.edu/~jburkardt/m_src/shallow_water_2d/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL;
using Timer;

namespace shallow_water
{
	public static class ShallowWater
    {
		public static void Main(string[] args)
		{
			if(args.Length != 2)
				throw new ArgumentException("Main() needs two arguments: setup-size and iterations");
			long size = Convert.ToInt64(args[0]);
			long iterations = Convert.ToInt64(args[1]);

			NumCIL.Bohrium.Utility.Activate();
			var data = new ShallowWaterSolver.DataDouble(size);
			using (new DispTimer(string.Format("ShallowWater {0}x{0}", size)))
                            ShallowWaterSolver.SolveDouble(iterations, data);
		}
    }
}
