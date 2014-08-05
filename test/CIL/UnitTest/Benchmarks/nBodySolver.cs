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

using R = NumCIL.Range;

using NumCIL.Double;
using DATA = System.Double;

namespace UnitTest.Benchmarks
{
	public static class nBodySolver
	{
		//Gravity
		private static DATA G = 1.0f;

		private static void FillDiagonal(NdArray a, DATA val)
		{
			long d  = a.Shape.Dimensions[0].Length;
			a.Reshape(new NumCIL.Shape(new long[] { d }, 0, new long[] { d+1 })).Set(val);
		}

		private static void CalcForce(Dictionary<string, NdArray> b)
		{
			var dx = b["x"] - b["x"][R.NewAxis, R.All].Transposed;
			FillDiagonal(dx, 1);
			var dy = b["y"] - b["y"][R.NewAxis, R.All].Transposed;
			FillDiagonal(dy, 1);
			var dz = b["z"] - b["z"][R.NewAxis, R.All].Transposed;
			FillDiagonal(dz, 1);
			var pm = b["m"] - b["m"][R.NewAxis, R.All].Transposed;
			FillDiagonal(pm, 0);

			var r = (dx.Pow(2) + dy.Pow(2) + dz.Pow(2)).Pow((DATA)0.5f);

			//In the below calc of the the forces the force of a body upon itself
			//becomes nan and thus destroys the data
			var Fx = G * pm / r.Pow(2) * (dx / r);
			var Fy = G * pm / r.Pow(2) * (dy / r);
			var Fz = G * pm / r.Pow(2) * (dz / r);

			//The diagonal nan numbers must be removed so that the force from a body
			//upon itself is zero
			FillDiagonal(Fx,0);
			FillDiagonal(Fy,0);
			FillDiagonal(Fz,0);

			b["vx"] += Add.Reduce(Fx, 1) / b["m"];
			b["vy"] += Add.Reduce(Fy, 1) / b["m"];
			b["vz"] += Add.Reduce(Fz, 1) / b["m"];
		}

		private static void Move(Dictionary<string, NdArray> galaxy)
		{
			CalcForce(galaxy);

			galaxy["x"] += galaxy["vx"];
			galaxy["y"] += galaxy["vy"];
			galaxy["z"] += galaxy["vz"];
		}

		private static Dictionary<string, NdArray> RandomGalaxy(long n, DATA xMax, DATA yMax, DATA zMax)
		{
			var res = new Dictionary<string, NdArray>();
            res["m"] = Generate.Random(n) * (DATA)Math.Pow(10, 6) / (DATA)(4 * Math.PI * Math.PI);
			res["x"] = Generate.Random(n) * 2 * xMax - xMax;
			res["y"] = Generate.Random(n) * 2 * yMax - yMax;
			res["z"] = Generate.Random(n) * 2 * zMax - zMax;
			res["vx"] = Generate.Zeroes(n);
			res["vy"] = Generate.Zeroes(n);
			res["vz"] = Generate.Zeroes(n);

			return res;
		}

		public static void Solve(long size, long steps)
		{
			var galaxy = RandomGalaxy(size, 500, 500, 500);
			for(long i = 0; i < steps; i++)
				Move(galaxy);
		}
	}
}

