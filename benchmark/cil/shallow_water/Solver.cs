

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
    public static class ShallowWaterSolver
    {
        /// <summary>
        /// NumCIL equivalent to [:]
        /// </summary>
        public static readonly R ALL = R.All;
        /// <summary>
        /// NumCIL equivalent to [1:-1]
        /// </summary>
        public static readonly R INNER = R.Slice(1, -1);
        /// <summary>
        /// NumCIL equivalent to [0]
        /// </summary>
        public static readonly R FIRST = R.El(0);
        /// <summary>
        /// NumCIL equivalent to [1]
        /// </summary>
        public static readonly R SECOND = R.El(1);
        /// <summary>
        /// NumCIL equivalent to [-1]
        /// </summary>
        public static readonly R LAST = R.El(-1);
        /// <summary>
        /// NumCIL equivalent to [:-1]
        /// </summary>
        public static readonly R ZM1 = R.Slice(0, -1);
        /// <summary>
        /// NumCIL equivalent to [1:]
        /// </summary>
        public static readonly R SKIP1 = R.Slice(1, 0);


        public class DataDouble
        {
            public readonly long N;
            public readonly NumCIL.Double.NdArray H;
            public readonly NumCIL.Double.NdArray U;
            public readonly NumCIL.Double.NdArray V;
            public readonly NumCIL.Double.NdArray Hx;
            public readonly NumCIL.Double.NdArray Ux;
            public readonly NumCIL.Double.NdArray Vx;
            public readonly NumCIL.Double.NdArray Hy;
            public readonly NumCIL.Double.NdArray Uy;
            public readonly NumCIL.Double.NdArray Vy;

            public DataDouble(long n)
            {
                N = n;
                H = NumCIL.Double.Generate.Ones(n + 2, n + 2);
                U = NumCIL.Double.Generate.Ones(n + 2, n + 2);
                V = NumCIL.Double.Generate.Ones(n + 2, n + 2);
                Hx = NumCIL.Double.Generate.Ones(n + 1, n + 1);
                   Ux = NumCIL.Double.Generate.Ones(n + 1, n + 1);
                Vx = NumCIL.Double.Generate.Ones(n + 1, n + 1);
                   Hy = NumCIL.Double.Generate.Ones(n + 1, n + 1);
                Uy = NumCIL.Double.Generate.Ones(n + 1, n + 1);
                Vy = NumCIL.Double.Generate.Ones(n + 1, n + 1);
            }
        }


        public static System.Double SolveDouble(long timesteps, DataDouble data)
        {
            //Convenience indices
            R RN = R.El(data.N); //Same as [n]
            R RN1 = R.El(data.N + 1); //Same as [n+1]

            System.Double g = 9.8f;// gravitational constant
            System.Double dt = 0.02f; // hardwired timestep
            System.Double dx = 1.0f;
            System.Double dy = 1.0f;
            long droploc = data.N / 4;

            var H = data.H;
            var U = data.U;
            var V = data.V;
            var Hx = data.Hx;
            var Ux = data.Ux;
            var Vx = data.Vx;
            var Hy = data.Hy;
            var Uy = data.Uy;
            var Vy = data.Vy;

            //Splash!!!
            H[droploc, droploc] += 5.0f;

            for (int i = 0; i < timesteps; i++)
            {
                H.Flush();
                // Reflecting boundary conditions
                H[ALL, FIRST] = H[ALL, SECOND];
                U[ALL, FIRST] = U[ALL, SECOND];
                V[ALL, FIRST] = -V[ALL, SECOND];
                H[ALL, RN1] = H[ALL, RN];
                U[ALL, RN1] = U[ALL, RN];
                V[ALL, RN1] = -V[ALL, RN];
                H[FIRST, ALL] = H[SECOND, ALL];
                U[FIRST, ALL] = -U[SECOND, ALL];
                V[FIRST, ALL] = V[SECOND, ALL];
                H[RN1, ALL] = H[RN, ALL];
                U[RN1, ALL] = -U[RN, ALL];
                V[RN1, ALL] = V[RN, ALL];

                //First half-step

                //Height
                Hx[ALL, R.Slice(0, -1)] = (H[SKIP1,INNER]+H[ZM1,INNER])/2 -
                        dt/(2*dx)*(U[SKIP1,INNER]-U[ZM1,INNER]);

                //x momentum
                Ux[ALL, R.Slice(0, -1)] = (U[SKIP1,INNER]+U[ZM1,INNER])/2 -
                dt/(2*dx)*((U[SKIP1,INNER].Pow(2)/H[SKIP1,INNER] +
                            g/2*H[SKIP1,INNER].Pow(2)) -
                           (U[ZM1,INNER].Pow(2)/H[ZM1,INNER] +
                            g/2*H[ZM1,INNER].Pow(2)));

                // y momentum
                Vx[ALL, ZM1] = (V[SKIP1,INNER]+V[ZM1,INNER])/2 -
                            dt/(2*dx)*((U[SKIP1,INNER] *
                            V[SKIP1,INNER]/H[SKIP1, INNER]) -
                           (U[ZM1,INNER] *
                            V[ZM1,INNER]/H[ZM1,INNER]));



                // height
                Hy[ZM1,ALL] = (H[INNER,SKIP1]+H[INNER,ZM1])/2 -
                            dt/(2*dy)*(V[INNER,SKIP1]-V[INNER,ZM1]);

                // x momentum
                Uy[ZM1,ALL] = (U[INNER,SKIP1]+U[INNER,ZM1])/2 -
                            dt/(2*dy)*((V[INNER,SKIP1] *
                            U[INNER,SKIP1]/H[INNER,SKIP1]) -
                           (V[INNER,ZM1] *
                            U[INNER,ZM1]/H[INNER,ZM1]));
                // y momentum
                Vy[ZM1,ALL] = (V[INNER,SKIP1]+V[INNER,ZM1])/2 -
                            dt/(2*dy)*((V[INNER,SKIP1].Pow(2)/H[INNER,SKIP1] +
                            g/2*H[INNER,SKIP1].Pow(2)) -
                           (V[INNER,ZM1].Pow(2)/H[INNER,ZM1] +
                            g/2*H[INNER,ZM1].Pow(2)));

                // Second half step

                // height
                H[INNER, INNER] = H[INNER, INNER] -
                            (dt/dx)*(Ux[SKIP1,ZM1]-Ux[ZM1,ZM1]) -
                            (dt/dy)*(Vy[ZM1,SKIP1]-Vy[ZM1, ZM1]);

                // x momentum
                U[INNER, INNER] = U[INNER, INNER] -
                           (dt/dx)*((Ux[SKIP1,ZM1].Pow(2)/Hx[SKIP1,ZM1] +
                             g/2*Hx[SKIP1,ZM1].Pow(2)) -
                            (Ux[ZM1,ZM1].Pow(2)/Hx[ZM1,ZM1] +
                             g / 2 * Hx[ZM1, ZM1].Pow(2))) -
                             (dt/dy)*((Vy[ZM1,SKIP1] *
                                       Uy[ZM1,SKIP1]/Hy[ZM1,SKIP1]) -
                                        (Vy[ZM1,ZM1] *
                                         Uy[ZM1,ZM1]/Hy[ZM1,ZM1]));
                // y momentum
                V[R.Slice(1, -1),R.Slice(1, -1)] = V[R.Slice(1, -1),R.Slice(1, -1)] -
                               (dt/dx)*((Ux[SKIP1,ZM1] *
                                         Vx[SKIP1,ZM1]/Hx[SKIP1,ZM1]) -
                                        (Ux[ZM1,ZM1]*Vx[ZM1,ZM1]/Hx[ZM1,ZM1])) -
                                        (dt / dy) * ((Vy[ZM1, SKIP1].Pow(2) / Hy[ZM1, SKIP1] +
                                                  g / 2 * Hy[ZM1, SKIP1].Pow(2)) -
                                                 (Vy[ZM1, ZM1].Pow(2) / Hy[ZM1, ZM1] +
                                                  g / 2 * Hy[ZM1, ZM1].Pow(2)));
            }
            //Make sure we have the actual data and use it as a checksum
            return NumCIL.Double.Add.Reduce(NumCIL.Double.Add.Reduce(H / data.N)).Value[0];
        }
        public class DataFloat
        {
            public readonly long N;
            public readonly NumCIL.Float.NdArray H;
            public readonly NumCIL.Float.NdArray U;
            public readonly NumCIL.Float.NdArray V;
            public readonly NumCIL.Float.NdArray Hx;
            public readonly NumCIL.Float.NdArray Ux;
            public readonly NumCIL.Float.NdArray Vx;
            public readonly NumCIL.Float.NdArray Hy;
            public readonly NumCIL.Float.NdArray Uy;
            public readonly NumCIL.Float.NdArray Vy;

            public DataFloat(long n)
            {
                N = n;
                H = NumCIL.Float.Generate.Ones(n + 2, n + 2);
                U = NumCIL.Float.Generate.Ones(n + 2, n + 2);
                V = NumCIL.Float.Generate.Ones(n + 2, n + 2);
                Hx = NumCIL.Float.Generate.Ones(n + 1, n + 1);
                   Ux = NumCIL.Float.Generate.Ones(n + 1, n + 1);
                Vx = NumCIL.Float.Generate.Ones(n + 1, n + 1);
                   Hy = NumCIL.Float.Generate.Ones(n + 1, n + 1);
                Uy = NumCIL.Float.Generate.Ones(n + 1, n + 1);
                Vy = NumCIL.Float.Generate.Ones(n + 1, n + 1);
            }
        }


        public static System.Single SolveFloat(long timesteps, DataFloat data)
        {
            //Convenience indices
            R RN = R.El(data.N); //Same as [n]
            R RN1 = R.El(data.N + 1); //Same as [n+1]

            System.Single g = 9.8f;// gravitational constant
            System.Single dt = 0.02f; // hardwired timestep
            System.Single dx = 1.0f;
            System.Single dy = 1.0f;
            long droploc = data.N / 4;

            var H = data.H;
            var U = data.U;
            var V = data.V;
            var Hx = data.Hx;
            var Ux = data.Ux;
            var Vx = data.Vx;
            var Hy = data.Hy;
            var Uy = data.Uy;
            var Vy = data.Vy;

            //Splash!!!
            H[droploc, droploc] += 5.0f;

            for (int i = 0; i < timesteps; i++)
            {
                H.Flush();
                // Reflecting boundary conditions
                H[ALL, FIRST] = H[ALL, SECOND];
                U[ALL, FIRST] = U[ALL, SECOND];
                V[ALL, FIRST] = -V[ALL, SECOND];
                H[ALL, RN1] = H[ALL, RN];
                U[ALL, RN1] = U[ALL, RN];
                V[ALL, RN1] = -V[ALL, RN];
                H[FIRST, ALL] = H[SECOND, ALL];
                U[FIRST, ALL] = -U[SECOND, ALL];
                V[FIRST, ALL] = V[SECOND, ALL];
                H[RN1, ALL] = H[RN, ALL];
                U[RN1, ALL] = -U[RN, ALL];
                V[RN1, ALL] = V[RN, ALL];

                //First half-step

                //Height
                Hx[ALL, R.Slice(0, -1)] = (H[SKIP1,INNER]+H[ZM1,INNER])/2 -
                        dt/(2*dx)*(U[SKIP1,INNER]-U[ZM1,INNER]);

                //x momentum
                Ux[ALL, R.Slice(0, -1)] = (U[SKIP1,INNER]+U[ZM1,INNER])/2 -
                dt/(2*dx)*((U[SKIP1,INNER].Pow(2)/H[SKIP1,INNER] +
                            g/2*H[SKIP1,INNER].Pow(2)) -
                           (U[ZM1,INNER].Pow(2)/H[ZM1,INNER] +
                            g/2*H[ZM1,INNER].Pow(2)));

                // y momentum
                Vx[ALL, ZM1] = (V[SKIP1,INNER]+V[ZM1,INNER])/2 -
                            dt/(2*dx)*((U[SKIP1,INNER] *
                            V[SKIP1,INNER]/H[SKIP1, INNER]) -
                           (U[ZM1,INNER] *
                            V[ZM1,INNER]/H[ZM1,INNER]));



                // height
                Hy[ZM1,ALL] = (H[INNER,SKIP1]+H[INNER,ZM1])/2 -
                            dt/(2*dy)*(V[INNER,SKIP1]-V[INNER,ZM1]);

                // x momentum
                Uy[ZM1,ALL] = (U[INNER,SKIP1]+U[INNER,ZM1])/2 -
                            dt/(2*dy)*((V[INNER,SKIP1] *
                            U[INNER,SKIP1]/H[INNER,SKIP1]) -
                           (V[INNER,ZM1] *
                            U[INNER,ZM1]/H[INNER,ZM1]));
                // y momentum
                Vy[ZM1,ALL] = (V[INNER,SKIP1]+V[INNER,ZM1])/2 -
                            dt/(2*dy)*((V[INNER,SKIP1].Pow(2)/H[INNER,SKIP1] +
                            g/2*H[INNER,SKIP1].Pow(2)) -
                           (V[INNER,ZM1].Pow(2)/H[INNER,ZM1] +
                            g/2*H[INNER,ZM1].Pow(2)));

                // Second half step

                // height
                H[INNER, INNER] = H[INNER, INNER] -
                            (dt/dx)*(Ux[SKIP1,ZM1]-Ux[ZM1,ZM1]) -
                            (dt/dy)*(Vy[ZM1,SKIP1]-Vy[ZM1, ZM1]);

                // x momentum
                U[INNER, INNER] = U[INNER, INNER] -
                           (dt/dx)*((Ux[SKIP1,ZM1].Pow(2)/Hx[SKIP1,ZM1] +
                             g/2*Hx[SKIP1,ZM1].Pow(2)) -
                            (Ux[ZM1,ZM1].Pow(2)/Hx[ZM1,ZM1] +
                             g / 2 * Hx[ZM1, ZM1].Pow(2))) -
                             (dt/dy)*((Vy[ZM1,SKIP1] *
                                       Uy[ZM1,SKIP1]/Hy[ZM1,SKIP1]) -
                                        (Vy[ZM1,ZM1] *
                                         Uy[ZM1,ZM1]/Hy[ZM1,ZM1]));
                // y momentum
                V[R.Slice(1, -1),R.Slice(1, -1)] = V[R.Slice(1, -1),R.Slice(1, -1)] -
                               (dt/dx)*((Ux[SKIP1,ZM1] *
                                         Vx[SKIP1,ZM1]/Hx[SKIP1,ZM1]) -
                                        (Ux[ZM1,ZM1]*Vx[ZM1,ZM1]/Hx[ZM1,ZM1])) -
                                        (dt / dy) * ((Vy[ZM1, SKIP1].Pow(2) / Hy[ZM1, SKIP1] +
                                                  g / 2 * Hy[ZM1, SKIP1].Pow(2)) -
                                                 (Vy[ZM1, ZM1].Pow(2) / Hy[ZM1, ZM1] +
                                                  g / 2 * Hy[ZM1, ZM1].Pow(2)));
            }
            //Make sure we have the actual data and use it as a checksum
            return NumCIL.Float.Add.Reduce(NumCIL.Float.Add.Reduce(H / data.N)).Value[0];
        }
    }
}
