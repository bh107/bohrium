
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

namespace nbody
{
	using NumCIL.Double;
	using DATA = System.Double;
	
	public static class nBodySolverDouble
	{
		//Gravity
		private static DATA G = (DATA)1.0;

		private static void FillDiagonal(NdArray a, DATA val)
		{
			long d  = a.Shape.Dimensions[0].Length;
			a.Reshape(new NumCIL.Shape(new long[] { d }, 0, new long[] { d+1 })).Set(val);
		}

		private static void CalcForce(Galaxy b)
		{
			var dx = b.x - b.x[R.NewAxis, R.All].Transposed;
			FillDiagonal(dx, 1);
			var dy = b.y - b.y[R.NewAxis, R.All].Transposed;
			FillDiagonal(dy, 1);
			var dz = b.z - b.z[R.NewAxis, R.All].Transposed;
			FillDiagonal(dz, 1);
			var pm = b.mass - b.mass[R.NewAxis, R.All].Transposed;
			FillDiagonal(pm, 0);

			var r = (dx.Pow(2) + dy.Pow(2) + dz.Pow(2)).Pow((DATA)0.5);

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

			b.vx += Add.Reduce(Fx, 1) / b.mass;
			b.vy += Add.Reduce(Fy, 1) / b.mass;
			b.vz += Add.Reduce(Fz, 1) / b.mass;
		}

		private static void Move(Galaxy galaxy)
		{
			CalcForce(galaxy);

			galaxy.x += galaxy.vx;
			galaxy.y += galaxy.vy;
			galaxy.z += galaxy.vz;
		}
		
		public class Galaxy
		{
			private const DATA XMAX = 500;
			private const DATA YMAX = 500;
			private const DATA ZMAX = 500;
			
			public NdArray mass;
			public NdArray x;
			public NdArray y;
			public NdArray z;
			public NdArray vx;
			public NdArray vy;
			public NdArray vz;
			
			public Galaxy(long size)
			{
				this.mass = Generate.Random(size) * (DATA)Math.Pow(10, 6) / (DATA)(4 * Math.PI * Math.PI);
				this.x = Generate.Random(size) * 2 * XMAX - XMAX;
				this.y = Generate.Random(size) * 2 * YMAX - YMAX;
				this.z = Generate.Random(size) * 2 * ZMAX - ZMAX;
				this.vx = Generate.Zeroes(size);
				this.vy = Generate.Zeroes(size);
				this.vz = Generate.Zeroes(size);			
			}
			
			public void Sync()
			{
				this.x.Flush();
				this.y.Flush();
				this.z.Flush();
			}
		}

		public static void Solve(Galaxy galaxy, long steps)
		{
			for(long i = 0; i < steps; i++)
				Move(galaxy);
		}
	}
}

namespace nbody
{
	using NumCIL.Double;
	using DATA = System.Double;
	
	public static class nBodySolverDoubleNoTempArrays
	{
		//Gravity
		private static DATA G = (DATA)1.0;

		private static void FillDiagonal(NdArray a, DATA val)
		{
			long d  = a.Shape.Dimensions[0].Length;
			a.Reshape(new NumCIL.Shape(new long[] { d }, 0, new long[] { d+1 })).Set(val);
		}

		private static void CalcForce(Galaxy b)
		{
			b.x.Transposed.Sub(b.x[R.NewAxis, R.All].Transposed, b.dx);
			FillDiagonal(b.dx, 1);
			b.y.Transposed.Sub(b.y[R.NewAxis, R.All].Transposed, b.dy);
			FillDiagonal(b.dy, 1);
			b.z.Transposed.Sub(b.z[R.NewAxis, R.All].Transposed, b.dz);
			FillDiagonal(b.dz, 1);
			b.mass.Transposed.Sub(b.mass[R.NewAxis, R.All].Transposed, b.pmass);
			FillDiagonal(b.pmass, 0);

			b.dx.Pow(2, b.r0);
			b.dx.Pow(2, b.r1);
			b.r0.Add(b.r1, b.r0);
			b.dz.Pow(2, b.r1);
			b.r0.Add(b.r1, b.r0);
			b.r0.Pow((DATA)0.5, b.r0);

			//In the below calc of the the forces the force of a body upon itself
			//becomes nan and thus destroys the data
			
			//Right-most divide, destroys dx,dy,dz
			b.dx.Div(b.r0, b.dx);
			b.dy.Div(b.r0, b.dy);
			b.dz.Div(b.r0, b.dz);

			//Combined pow for r
			b.r0.Pow(2, b.r0);
			
			//Combined multiply
			b.pmass.Mul(G, b.r1);
			
			//Combined div
			b.r1.Div(b.r0, b.r1);
			
			//Update outputs, use dx,dy,dz for storing
			b.dx.Mul(b.r1, b.dx);
			b.dy.Mul(b.r1, b.dy);
			b.dz.Mul(b.r1, b.dz);
			
			//The diagonal nan numbers must be removed so that the force from a body
			//upon itself is zero
			FillDiagonal(b.dx, 0);
			FillDiagonal(b.dy, 0);
			FillDiagonal(b.dz, 0);

			b.dx.Reduce<Add>(1, b.t0);
			b.t0.Div(b.mass, b.t0);
			b.vx.Add(b.t0, b.vx);
			b.dy.Reduce<Add>(1, b.t0);
			b.t0.Div(b.mass, b.t0);
			b.vy.Add(b.t0, b.vx);
			b.dz.Reduce<Add>(1, b.t0);
			b.t0.Div(b.mass, b.t0);
			b.vz.Add(b.t0, b.vx);
		}

		private static void Move(Galaxy galaxy)
		{
			CalcForce(galaxy);
			
			galaxy.x.Add(galaxy.vx, galaxy.x);
			galaxy.y.Add(galaxy.vy, galaxy.y);
			galaxy.z.Add(galaxy.vz, galaxy.z);
		}
		
		public class Galaxy
		{
			private const DATA XMAX = 500;
			private const DATA YMAX = 500;
			private const DATA ZMAX = 500;
			
			public NdArray mass;
			public NdArray x;
			public NdArray y;
			public NdArray z;
			public NdArray vx;
			public NdArray vy;
			public NdArray vz;
			public NdArray dx;
			public NdArray dy;
			public NdArray dz;
			public NdArray pmass;
			public NdArray r0;
			public NdArray r1;
			public NdArray t0;
			
			public Galaxy(long size)
			{
				this.mass = Generate.Random(size) * (DATA)Math.Pow(10, 6) / (DATA)(4 * Math.PI * Math.PI);
				this.x = Generate.Random(size) * 2 * XMAX - XMAX;
				this.y = Generate.Random(size) * 2 * YMAX - YMAX;
				this.z = Generate.Random(size) * 2 * ZMAX - ZMAX;
				this.vx = Generate.Zeroes(size);
				this.vy = Generate.Zeroes(size);
				this.vz = Generate.Zeroes(size);
				this.dx = new NdArray(new [] {size, size});
				this.dy = new NdArray(new [] {size, size});
				this.dz = new NdArray(new [] {size, size});
				this.pmass = new NdArray(new [] {size, size});
				this.r0 = new NdArray(this.dx.Shape);
				this.r1 = new NdArray(this.dx.Shape);
				this.t0 = new NdArray(this.x.Shape);
			}
			
			public void Sync()
			{
				this.x.Flush();
				this.y.Flush();
				this.z.Flush();
			}
		}

		public static void Solve(Galaxy galaxy, long steps)
		{
			for(long i = 0; i < steps; i++)
				Move(galaxy);
		}
	}
}

namespace nbody
{
	using NumCIL.Float;
	using DATA = System.Single;
	
	public static class nBodySolverFloat
	{
		//Gravity
		private static DATA G = (DATA)1.0;

		private static void FillDiagonal(NdArray a, DATA val)
		{
			long d  = a.Shape.Dimensions[0].Length;
			a.Reshape(new NumCIL.Shape(new long[] { d }, 0, new long[] { d+1 })).Set(val);
		}

		private static void CalcForce(Galaxy b)
		{
			var dx = b.x - b.x[R.NewAxis, R.All].Transposed;
			FillDiagonal(dx, 1);
			var dy = b.y - b.y[R.NewAxis, R.All].Transposed;
			FillDiagonal(dy, 1);
			var dz = b.z - b.z[R.NewAxis, R.All].Transposed;
			FillDiagonal(dz, 1);
			var pm = b.mass - b.mass[R.NewAxis, R.All].Transposed;
			FillDiagonal(pm, 0);

			var r = (dx.Pow(2) + dy.Pow(2) + dz.Pow(2)).Pow((DATA)0.5);

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

			b.vx += Add.Reduce(Fx, 1) / b.mass;
			b.vy += Add.Reduce(Fy, 1) / b.mass;
			b.vz += Add.Reduce(Fz, 1) / b.mass;
		}

		private static void Move(Galaxy galaxy)
		{
			CalcForce(galaxy);

			galaxy.x += galaxy.vx;
			galaxy.y += galaxy.vy;
			galaxy.z += galaxy.vz;
		}
		
		public class Galaxy
		{
			private const DATA XMAX = 500;
			private const DATA YMAX = 500;
			private const DATA ZMAX = 500;
			
			public NdArray mass;
			public NdArray x;
			public NdArray y;
			public NdArray z;
			public NdArray vx;
			public NdArray vy;
			public NdArray vz;
			
			public Galaxy(long size)
			{
				this.mass = Generate.Random(size) * (DATA)Math.Pow(10, 6) / (DATA)(4 * Math.PI * Math.PI);
				this.x = Generate.Random(size) * 2 * XMAX - XMAX;
				this.y = Generate.Random(size) * 2 * YMAX - YMAX;
				this.z = Generate.Random(size) * 2 * ZMAX - ZMAX;
				this.vx = Generate.Zeroes(size);
				this.vy = Generate.Zeroes(size);
				this.vz = Generate.Zeroes(size);			
			}
			
			public void Sync()
			{
				this.x.Flush();
				this.y.Flush();
				this.z.Flush();
			}
		}

		public static void Solve(Galaxy galaxy, long steps)
		{
			for(long i = 0; i < steps; i++)
				Move(galaxy);
		}
	}
}

namespace nbody
{
	using NumCIL.Float;
	using DATA = System.Single;
	
	public static class nBodySolverFloatNoTempArrays
	{
		//Gravity
		private static DATA G = (DATA)1.0;

		private static void FillDiagonal(NdArray a, DATA val)
		{
			long d  = a.Shape.Dimensions[0].Length;
			a.Reshape(new NumCIL.Shape(new long[] { d }, 0, new long[] { d+1 })).Set(val);
		}

		private static void CalcForce(Galaxy b)
		{
			b.x.Transposed.Sub(b.x[R.NewAxis, R.All].Transposed, b.dx);
			FillDiagonal(b.dx, 1);
			b.y.Transposed.Sub(b.y[R.NewAxis, R.All].Transposed, b.dy);
			FillDiagonal(b.dy, 1);
			b.z.Transposed.Sub(b.z[R.NewAxis, R.All].Transposed, b.dz);
			FillDiagonal(b.dz, 1);
			b.mass.Transposed.Sub(b.mass[R.NewAxis, R.All].Transposed, b.pmass);
			FillDiagonal(b.pmass, 0);

			b.dx.Pow(2, b.r0);
			b.dx.Pow(2, b.r1);
			b.r0.Add(b.r1, b.r0);
			b.dz.Pow(2, b.r1);
			b.r0.Add(b.r1, b.r0);
			b.r0.Pow((DATA)0.5, b.r0);

			//In the below calc of the the forces the force of a body upon itself
			//becomes nan and thus destroys the data
			
			//Right-most divide, destroys dx,dy,dz
			b.dx.Div(b.r0, b.dx);
			b.dy.Div(b.r0, b.dy);
			b.dz.Div(b.r0, b.dz);

			//Combined pow for r
			b.r0.Pow(2, b.r0);
			
			//Combined multiply
			b.pmass.Mul(G, b.r1);
			
			//Combined div
			b.r1.Div(b.r0, b.r1);
			
			//Update outputs, use dx,dy,dz for storing
			b.dx.Mul(b.r1, b.dx);
			b.dy.Mul(b.r1, b.dy);
			b.dz.Mul(b.r1, b.dz);
			
			//The diagonal nan numbers must be removed so that the force from a body
			//upon itself is zero
			FillDiagonal(b.dx, 0);
			FillDiagonal(b.dy, 0);
			FillDiagonal(b.dz, 0);

			b.dx.Reduce<Add>(1, b.t0);
			b.t0.Div(b.mass, b.t0);
			b.vx.Add(b.t0, b.vx);
			b.dy.Reduce<Add>(1, b.t0);
			b.t0.Div(b.mass, b.t0);
			b.vy.Add(b.t0, b.vx);
			b.dz.Reduce<Add>(1, b.t0);
			b.t0.Div(b.mass, b.t0);
			b.vz.Add(b.t0, b.vx);
		}

		private static void Move(Galaxy galaxy)
		{
			CalcForce(galaxy);
			
			galaxy.x.Add(galaxy.vx, galaxy.x);
			galaxy.y.Add(galaxy.vy, galaxy.y);
			galaxy.z.Add(galaxy.vz, galaxy.z);
		}
		
		public class Galaxy
		{
			private const DATA XMAX = 500;
			private const DATA YMAX = 500;
			private const DATA ZMAX = 500;
			
			public NdArray mass;
			public NdArray x;
			public NdArray y;
			public NdArray z;
			public NdArray vx;
			public NdArray vy;
			public NdArray vz;
			public NdArray dx;
			public NdArray dy;
			public NdArray dz;
			public NdArray pmass;
			public NdArray r0;
			public NdArray r1;
			public NdArray t0;
			
			public Galaxy(long size)
			{
				this.mass = Generate.Random(size) * (DATA)Math.Pow(10, 6) / (DATA)(4 * Math.PI * Math.PI);
				this.x = Generate.Random(size) * 2 * XMAX - XMAX;
				this.y = Generate.Random(size) * 2 * YMAX - YMAX;
				this.z = Generate.Random(size) * 2 * ZMAX - ZMAX;
				this.vx = Generate.Zeroes(size);
				this.vy = Generate.Zeroes(size);
				this.vz = Generate.Zeroes(size);
				this.dx = new NdArray(new [] {size, size});
				this.dy = new NdArray(new [] {size, size});
				this.dz = new NdArray(new [] {size, size});
				this.pmass = new NdArray(new [] {size, size});
				this.r0 = new NdArray(this.dx.Shape);
				this.r1 = new NdArray(this.dx.Shape);
				this.t0 = new NdArray(this.x.Shape);
			}
			
			public void Sync()
			{
				this.x.Flush();
				this.y.Flush();
				this.z.Flush();
			}
		}

		public static void Solve(Galaxy galaxy, long steps)
		{
			for(long i = 0; i < steps; i++)
				Move(galaxy);
		}
	}
}

