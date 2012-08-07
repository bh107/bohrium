using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using R = NumCIL.Range;

using NumCIL.Double;
using DATA = System.Double;

namespace Tester
{
	public static class nBodySolver
	{
		//Gravity
		private static DATA G = 1.0;

		private static void FillDiagonal(NdArray a, DATA val)
		{
			long d  = a.Shape.Dimensions[0].Length;
			a.Reshape(new NumCIL.Shape(new long[] { d }, 0, new long[] { d+1 })).Set(val);
		}

		private static void CalcForce(Dictionary<string, NdArray> b)
		{
			var dx = b["x"] - b["x"][R.NewAxis, R.All].Transposed;
			FillDiagonal(dx, 1.0);
			var dy = b["y"] - b["y"][R.NewAxis, R.All].Transposed;
			FillDiagonal(dy, 1.0);
			var dz = b["z"] - b["z"][R.NewAxis, R.All].Transposed;
			FillDiagonal(dz, 1.0);
			var pm = b["m"] - b["m"][R.NewAxis, R.All].Transposed;
			FillDiagonal(pm, 0.0);

			var r = (dx.Pow(2) + dy.Pow(2) + dz.Pow(2)).Pow(0.5);

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

		private static Dictionary<string, NdArray> RandomGalaxy(long n, DATA x_max, DATA y_max, DATA z_max)
		{
			Dictionary<string, NdArray> res = new Dictionary<string, NdArray>();
			res["m"] = Generate.Random(n) * Math.Pow(10, 6) / (4 * Math.PI * Math.PI);
			res["x"] = Generate.Random(n) * 2 * x_max - x_max;
			res["y"] = Generate.Random(n) * 2 * y_max - y_max;
			res["z"] = Generate.Random(n) * 2 * z_max - z_max;
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

