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
using Utilities;

namespace shallow_water
{
	public static class ShallowWater
    {
		public static void Main (string[] args)
		{
			Dictionary<string, string> dict = CommandLineParser.ExtractOptions (args.ToList ());
			if (!dict.ContainsKey ("size"))
				throw new ArgumentException ("Main() needs the size argument (fx --size=1000*1000*10)");
			string size = dict ["size"];
			string dtype = "";
			if (dict.ContainsKey ("dtype"))
				dtype = dict ["dtype"];

			string[] sizes = size.Split ('*');
			if(sizes.Length != 3)
				throw new ArgumentException ("The size argument must consist of three dimensions (fx --size=1000*1000*10)");

			long sizeX = Convert.ToInt64 (sizes [0]);
			long sizeY = Convert.ToInt64 (sizes [1]);
			long iterations = Convert.ToInt64 (sizes [2]);

			if(sizeX != sizeY)
				throw new ArgumentException ("The two dimension arguments must be identical (fx --size=1000*1000*10)");

			NumCIL.Bohrium.Utility.Activate ();
			if (dtype.StartsWith ("D", StringComparison.OrdinalIgnoreCase)) {
				var data = new ShallowWaterSolver.DataDouble (sizeX);
				using (new DispTimer(string.Format("ShallowWater (Double) {0}*{1}*{2}", sizeX, sizeY, iterations)))
					ShallowWaterSolver.SolveDouble (iterations, data);
			} else {
				var data = new ShallowWaterSolver.DataFloat (sizeX);
				using (new DispTimer(string.Format("ShallowWater (Float) {0}*{1}*{2}", sizeX, sizeY, iterations)))
					ShallowWaterSolver.SolveFloat (iterations, data);
			}
		}
    }
}
