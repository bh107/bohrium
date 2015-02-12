
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

namespace nice
{

	using NumCIL.Double;
	using TArray = NumCIL.Double.NdArray;
	using TData = System.Double;

    public static class NiceSolverDouble
    {
		//private static readonly Utilities.Generator<TArray, TData> Generate = new Utilities.Generator<TArray, TData>();

		private static TData ConvertValue(object o) { return (TData)o; }

		//Gravity
		private static TData G = (TData)6.673e-11;

		// Solar mass
		public const TData SOLARMASS = (TData)1.98892e30;

		// Discrete Time units
		public const TData DT = (TData)1e12;


		private static void FillDiagonal(TArray a, TData val)
		{
			long d  = a.Shape.Dimensions[0].Length;
			a.Reshape(new NumCIL.Shape(new long[] { d }, 0, new long[] { d+1 })).Set(val);
		}

		private static void CalcForce(Bodies a, Bodies b)
		{
			var dx = b.x - a.x[R.NewAxis, R.All].Transposed;
			var dy = b.y - a.y[R.NewAxis, R.All].Transposed;
			var dz = b.z - a.z[R.NewAxis, R.All].Transposed;
			var pm = b.mass * a.mass[R.NewAxis, R.All].Transposed;

			if (a == b)
			{
				FillDiagonal(dx, 1);
				FillDiagonal(dy, 1);
				FillDiagonal(dz, 1);
				FillDiagonal(pm, 0);
			}

			var r = (dx.Pow(2) + dy.Pow(2) + dz.Pow(2)).Pow((TData)0.5);

			//In the below calc of the the forces the force of a body upon itself
			//becomes nan and thus destroys the data
			var Fx = G * pm / r.Pow(2) * (dx / r);
			var Fy = G * pm / r.Pow(2) * (dy / r);
			var Fz = G * pm / r.Pow(2) * (dz / r);

			//The diagonal nan numbers must be removed so that the force from a body
			//upon itself is zero
			if (a == b)
			{
				FillDiagonal(Fx, 0);
				FillDiagonal(Fy, 0);
				FillDiagonal(Fz, 0);
			}

			a.vx += Add.Reduce(Fx, 1) / a.mass * DT;
			a.vy += Add.Reduce(Fy, 1) / a.mass * DT;
			a.vz += Add.Reduce(Fz, 1) / a.mass * DT;
		}

		private static void Move(Galaxy galaxy)
		{
			CalcForce(galaxy.SolarSystem, galaxy.SolarSystem);
			CalcForce(galaxy.Asteroids, galaxy.SolarSystem);

			galaxy.SolarSystem.x += galaxy.SolarSystem.vx * DT;
			galaxy.SolarSystem.y += galaxy.SolarSystem.vy * DT;
			galaxy.SolarSystem.z += galaxy.SolarSystem.vz * DT;

			galaxy.Asteroids.x += galaxy.Asteroids.vx * DT;
			galaxy.Asteroids.y += galaxy.Asteroids.vy * DT;
			galaxy.Asteroids.z += galaxy.Asteroids.vz * DT;
		}


		public class Galaxy
		{
			public Bodies SolarSystem;
			public Bodies Asteroids;

			public Galaxy(long planets, long asteroids)
			{
				this.SolarSystem = new SolarSystem(planets);
				this.Asteroids = new Asteroids(asteroids);
			}
		}

		public abstract class Bodies
		{
			public const TData XMAX = (TData)1e18;
			public const TData YMAX = (TData)1e18;
			public const TData ZMAX = (TData)1e18;

			public TArray mass;
			public TArray x;
			public TArray y;
			public TArray z;
			public TArray vx;
			public TArray vy;
			public TArray vz;

			protected void Reset(long size)
			{
				this.x = Generate.Random(size);
				this.y = Generate.Random(size);
				this.z = Generate.Random(size) * (TData)0.01;

				var dist = 1f / Sqrt.Apply((this.x.Pow(2f) + this.y.Pow(2f) + this.z.Pow(2f)));
				dist = dist - (TData)(0.8f - (new Random().NextDouble() * 0.1f));

				this.x = XMAX * this.x * dist * Sign.Apply(.5f - Generate.Random(size));
				this.y = YMAX * this.y * dist * Sign.Apply(.5f - Generate.Random(size));
				this.z = ZMAX * this.z * dist * Sign.Apply(.5f - Generate.Random(size));

				var magv = Cirklev(this.x, this.y, this.z);
				var absangle = Atan.Apply(Abs.Apply(this.y / this.x));
				var thetav= (TData)(Math.PI/2) - absangle;

				this.vx = (TData)(-1) * Sign.Apply(this.y) * Cos.Apply(thetav) * magv;
				this.vy = Sign.Apply(this.x) * Sin.Apply(thetav) * magv;
				this.vz = Generate.Zeroes(size);
			}

			private TArray Cirklev(TArray rx, TArray ry, TArray rz)
			{
				var r2 = Sqrt.Apply(rx * rx + ry * ry + rz * rz);
				var numerator = (TData)((6.67e-11) * 1e6 * SOLARMASS);
				return Sqrt.Apply(numerator / r2);
			}
		}

		public class SolarSystem : Bodies
		{
			public SolarSystem(long size)
			{
				base.Reset(size);

				this.mass = (Generate.Random(size) * (TData)(SOLARMASS * 10)) + (TData)1e20;

				this.mass[0]= (TData)1e6 * SOLARMASS;
				this.x[0]= 0;
				this.y[0]= 0;
				this.z[0]= 0;
				this.vx[0]= 0;
				this.vy[0]= 0;
				this.vz[0]= 0;
			}
		}


		public class Asteroids : Bodies
		{
			public Asteroids(long size)
			{
				base.Reset(size);
				this.mass = (Generate.Random(size) * (TData)(SOLARMASS * 10)) + (TData)1e14;
			}
		}

		public static Galaxy Create(long planets, long asteroids)
		{
			return new Galaxy(planets, asteroids);
		}

		public static void Solve(Galaxy galaxy, long steps, bool image_output)
		{
			if (image_output)
				Render(galaxy, 0, steps);

			for (long step = 0; step < steps; step++)
			{
				Move(galaxy);

				if (image_output)
					Render(galaxy, step + 1, steps);
			}
		}

		private static void Render(Galaxy galaxy, long step, long steps)
		{
			var image_width = 1024;
			var image_height = 1024;

			var axz = galaxy.Asteroids.x;// / galaxy.Asteroids.z;
			var ayz = galaxy.Asteroids.y;// / galaxy.Asteroids.z;
			var sxz = galaxy.SolarSystem.x;// / galaxy.SolarSystem.z;
			var syz = galaxy.SolarSystem.y;// / galaxy.SolarSystem.z;

			Func<int, int, int, System.Drawing.Rectangle> inflateBox = (x, y, s) => new System.Drawing.Rectangle(new System.Drawing.Point(x - (s/2), y - (s/2)), new System.Drawing.Size(s, s));

			var mass_min = Min.Reduce(galaxy.SolarSystem.mass).Value[0];
			var mass_max = Max.Reduce(galaxy.SolarSystem.mass).Value[0];

			var rangex_max = Math.Max(Max.Reduce(axz).Value[0], Max.Reduce(sxz).Value[0]);
			var rangex_min = Math.Min(Min.Reduce(axz).Value[0], Min.Reduce(sxz).Value[0]);

			var rangey_max = Math.Max(Max.Reduce(ayz).Value[0], Max.Reduce(syz).Value[0]);
			var rangey_min = Math.Min(Min.Reduce(ayz).Value[0], Min.Reduce(syz).Value[0]);

			var rangex_diff = rangex_max - rangex_min;
			var rangey_diff = rangey_max - rangey_min;

			rangex_min = -Bodies.XMAX * 1.2f;
			rangey_min = -Bodies.YMAX * 1.2f;
			rangex_diff = Bodies.XMAX * 2.4f;
			rangey_diff = Bodies.YMAX * 2.4f;

			var sx = NumCIL.Int32.Min.Apply(image_width - 1, NumCIL.Int32.Max.Apply(0, ((NumCIL.Int32.NdArray)(((sxz - rangex_min) / rangex_diff) * (image_width - 1))))).AsArray();
			var sy = NumCIL.Int32.Min.Apply(image_height - 1, NumCIL.Int32.Max.Apply(0, ((NumCIL.Int32.NdArray)(((syz - rangey_min) / rangey_diff) * (image_height - 1))))).AsArray();

			var ax = NumCIL.Int32.Min.Apply(image_width - 1, NumCIL.Int32.Max.Apply(0, ((NumCIL.Int32.NdArray)(((axz - rangex_min) / rangex_diff) * (image_width - 1))))).AsArray();
			var ay = NumCIL.Int32.Min.Apply(image_height - 1, NumCIL.Int32.Max.Apply(0, ((NumCIL.Int32.NdArray)(((ayz - rangey_min) / rangey_diff) * (image_height - 1))))).AsArray();

			using (var image = new System.Drawing.Bitmap(image_width, image_height))
			{
				using (var gc = System.Drawing.Graphics.FromImage(image))
					gc.Clear(System.Drawing.Color.White);
					
				using (var gc = System.Drawing.Graphics.FromImage(image))
				{
					for (var s = 0; s < ax.Length; s++)
						gc.DrawEllipse(System.Drawing.Pens.DarkGray, inflateBox(ax[s], ay[s], 1));
					for (var s = 0; s < sx.Length; s++)
						gc.DrawEllipse(System.Drawing.Pens.Red, inflateBox(sx[s], sy[s], 2));
				}

				image.Save(string.Format("step-{0:0000}.png", step), System.Drawing.Imaging.ImageFormat.Png);
			}

			Console.WriteLine("Completed step {0} of {1}", step, steps);
		}
	}
}

namespace nice
{

	using NumCIL.Float;
	using TArray = NumCIL.Float.NdArray;
	using TData = System.Single;

    public static class NiceSolverSingle
    {
		//private static readonly Utilities.Generator<TArray, TData> Generate = new Utilities.Generator<TArray, TData>();

		private static TData ConvertValue(object o) { return (TData)o; }

		//Gravity
		private static TData G = (TData)6.673e-11;

		// Solar mass
		public const TData SOLARMASS = (TData)1.98892e30;

		// Discrete Time units
		public const TData DT = (TData)1e12;


		private static void FillDiagonal(TArray a, TData val)
		{
			long d  = a.Shape.Dimensions[0].Length;
			a.Reshape(new NumCIL.Shape(new long[] { d }, 0, new long[] { d+1 })).Set(val);
		}

		private static void CalcForce(Bodies a, Bodies b)
		{
			var dx = b.x - a.x[R.NewAxis, R.All].Transposed;
			var dy = b.y - a.y[R.NewAxis, R.All].Transposed;
			var dz = b.z - a.z[R.NewAxis, R.All].Transposed;
			var pm = b.mass * a.mass[R.NewAxis, R.All].Transposed;

			if (a == b)
			{
				FillDiagonal(dx, 1);
				FillDiagonal(dy, 1);
				FillDiagonal(dz, 1);
				FillDiagonal(pm, 0);
			}

			var r = (dx.Pow(2) + dy.Pow(2) + dz.Pow(2)).Pow((TData)0.5);

			//In the below calc of the the forces the force of a body upon itself
			//becomes nan and thus destroys the data
			var Fx = G * pm / r.Pow(2) * (dx / r);
			var Fy = G * pm / r.Pow(2) * (dy / r);
			var Fz = G * pm / r.Pow(2) * (dz / r);

			//The diagonal nan numbers must be removed so that the force from a body
			//upon itself is zero
			if (a == b)
			{
				FillDiagonal(Fx, 0);
				FillDiagonal(Fy, 0);
				FillDiagonal(Fz, 0);
			}

			a.vx += Add.Reduce(Fx, 1) / a.mass * DT;
			a.vy += Add.Reduce(Fy, 1) / a.mass * DT;
			a.vz += Add.Reduce(Fz, 1) / a.mass * DT;
		}

		private static void Move(Galaxy galaxy)
		{
			CalcForce(galaxy.SolarSystem, galaxy.SolarSystem);
			CalcForce(galaxy.Asteroids, galaxy.SolarSystem);

			galaxy.SolarSystem.x += galaxy.SolarSystem.vx * DT;
			galaxy.SolarSystem.y += galaxy.SolarSystem.vy * DT;
			galaxy.SolarSystem.z += galaxy.SolarSystem.vz * DT;

			galaxy.Asteroids.x += galaxy.Asteroids.vx * DT;
			galaxy.Asteroids.y += galaxy.Asteroids.vy * DT;
			galaxy.Asteroids.z += galaxy.Asteroids.vz * DT;
		}


		public class Galaxy
		{
			public Bodies SolarSystem;
			public Bodies Asteroids;

			public Galaxy(long planets, long asteroids)
			{
				this.SolarSystem = new SolarSystem(planets);
				this.Asteroids = new Asteroids(asteroids);
			}
		}

		public abstract class Bodies
		{
			public const TData XMAX = (TData)1e18;
			public const TData YMAX = (TData)1e18;
			public const TData ZMAX = (TData)1e18;

			public TArray mass;
			public TArray x;
			public TArray y;
			public TArray z;
			public TArray vx;
			public TArray vy;
			public TArray vz;

			protected void Reset(long size)
			{
				this.x = Generate.Random(size);
				this.y = Generate.Random(size);
				this.z = Generate.Random(size) * (TData)0.01;

				var dist = 1f / Sqrt.Apply((this.x.Pow(2f) + this.y.Pow(2f) + this.z.Pow(2f)));
				dist = dist - (TData)(0.8f - (new Random().NextDouble() * 0.1f));

				this.x = XMAX * this.x * dist * Sign.Apply(.5f - Generate.Random(size));
				this.y = YMAX * this.y * dist * Sign.Apply(.5f - Generate.Random(size));
				this.z = ZMAX * this.z * dist * Sign.Apply(.5f - Generate.Random(size));

				var magv = Cirklev(this.x, this.y, this.z);
				var absangle = Atan.Apply(Abs.Apply(this.y / this.x));
				var thetav= (TData)(Math.PI/2) - absangle;

				this.vx = (TData)(-1) * Sign.Apply(this.y) * Cos.Apply(thetav) * magv;
				this.vy = Sign.Apply(this.x) * Sin.Apply(thetav) * magv;
				this.vz = Generate.Zeroes(size);
			}

			private TArray Cirklev(TArray rx, TArray ry, TArray rz)
			{
				var r2 = Sqrt.Apply(rx * rx + ry * ry + rz * rz);
				var numerator = (TData)((6.67e-11) * 1e6 * SOLARMASS);
				return Sqrt.Apply(numerator / r2);
			}
		}

		public class SolarSystem : Bodies
		{
			public SolarSystem(long size)
			{
				base.Reset(size);

				this.mass = (Generate.Random(size) * (TData)(SOLARMASS * 10)) + (TData)1e20;

				this.mass[0]= (TData)1e6 * SOLARMASS;
				this.x[0]= 0;
				this.y[0]= 0;
				this.z[0]= 0;
				this.vx[0]= 0;
				this.vy[0]= 0;
				this.vz[0]= 0;
			}
		}


		public class Asteroids : Bodies
		{
			public Asteroids(long size)
			{
				base.Reset(size);
				this.mass = (Generate.Random(size) * (TData)(SOLARMASS * 10)) + (TData)1e14;
			}
		}

		public static Galaxy Create(long planets, long asteroids)
		{
			return new Galaxy(planets, asteroids);
		}

		public static void Solve(Galaxy galaxy, long steps, bool image_output)
		{
			if (image_output)
				Render(galaxy, 0, steps);

			for (long step = 0; step < steps; step++)
			{
				Move(galaxy);

				if (image_output)
					Render(galaxy, step + 1, steps);
			}
		}

		private static void Render(Galaxy galaxy, long step, long steps)
		{
			var image_width = 1024;
			var image_height = 1024;

			var axz = galaxy.Asteroids.x;// / galaxy.Asteroids.z;
			var ayz = galaxy.Asteroids.y;// / galaxy.Asteroids.z;
			var sxz = galaxy.SolarSystem.x;// / galaxy.SolarSystem.z;
			var syz = galaxy.SolarSystem.y;// / galaxy.SolarSystem.z;

			Func<int, int, int, System.Drawing.Rectangle> inflateBox = (x, y, s) => new System.Drawing.Rectangle(new System.Drawing.Point(x - (s/2), y - (s/2)), new System.Drawing.Size(s, s));

			var mass_min = Min.Reduce(galaxy.SolarSystem.mass).Value[0];
			var mass_max = Max.Reduce(galaxy.SolarSystem.mass).Value[0];

			var rangex_max = Math.Max(Max.Reduce(axz).Value[0], Max.Reduce(sxz).Value[0]);
			var rangex_min = Math.Min(Min.Reduce(axz).Value[0], Min.Reduce(sxz).Value[0]);

			var rangey_max = Math.Max(Max.Reduce(ayz).Value[0], Max.Reduce(syz).Value[0]);
			var rangey_min = Math.Min(Min.Reduce(ayz).Value[0], Min.Reduce(syz).Value[0]);

			var rangex_diff = rangex_max - rangex_min;
			var rangey_diff = rangey_max - rangey_min;

			rangex_min = -Bodies.XMAX * 1.2f;
			rangey_min = -Bodies.YMAX * 1.2f;
			rangex_diff = Bodies.XMAX * 2.4f;
			rangey_diff = Bodies.YMAX * 2.4f;

			var sx = NumCIL.Int32.Min.Apply(image_width - 1, NumCIL.Int32.Max.Apply(0, ((NumCIL.Int32.NdArray)(((sxz - rangex_min) / rangex_diff) * (image_width - 1))))).AsArray();
			var sy = NumCIL.Int32.Min.Apply(image_height - 1, NumCIL.Int32.Max.Apply(0, ((NumCIL.Int32.NdArray)(((syz - rangey_min) / rangey_diff) * (image_height - 1))))).AsArray();

			var ax = NumCIL.Int32.Min.Apply(image_width - 1, NumCIL.Int32.Max.Apply(0, ((NumCIL.Int32.NdArray)(((axz - rangex_min) / rangex_diff) * (image_width - 1))))).AsArray();
			var ay = NumCIL.Int32.Min.Apply(image_height - 1, NumCIL.Int32.Max.Apply(0, ((NumCIL.Int32.NdArray)(((ayz - rangey_min) / rangey_diff) * (image_height - 1))))).AsArray();

			using (var image = new System.Drawing.Bitmap(image_width, image_height))
			{
				using (var gc = System.Drawing.Graphics.FromImage(image))
					gc.Clear(System.Drawing.Color.White);
					
				using (var gc = System.Drawing.Graphics.FromImage(image))
				{
					for (var s = 0; s < ax.Length; s++)
						gc.DrawEllipse(System.Drawing.Pens.DarkGray, inflateBox(ax[s], ay[s], 1));
					for (var s = 0; s < sx.Length; s++)
						gc.DrawEllipse(System.Drawing.Pens.Red, inflateBox(sx[s], sy[s], 2));
				}

				image.Save(string.Format("step-{0:0000}.png", step), System.Drawing.Imaging.ImageFormat.Png);
			}

			Console.WriteLine("Completed step {0} of {1}", step, steps);
		}
	}
}

