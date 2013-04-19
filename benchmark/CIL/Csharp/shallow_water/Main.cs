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
using R = NumCIL.Range;
using NumCIL;

namespace shallow_water
{
	public static class ShallowWater
    {
		public static void Main(string[] args)
		{
			NumCIL.Bohrium.Utility.Activate();
			var data = new ShallowWaterSolver.DataDouble(10);
			ShallowWaterSolver.SolveDouble(10, data);
		}
    }
}
