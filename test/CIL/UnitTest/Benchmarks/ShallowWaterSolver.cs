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
using NumCIL.Double;
using T = System.Double;
using R = NumCIL.Range;
using NumCIL;

namespace UnitTest.Benchmarks
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

        public static T Solve(long n, long timesteps)
        {
            //Convenience indices
            R RN = R.El(n); //Same as [n]
            R RN1 = R.El(n + 1); //Same as [n+1]

            T g = 9.8f;// gravitational constant
            T dt = 0.02f; // hardwired timestep
            T dx = 1.0f;
            T dy = 1.0f;    
            long droploc = n / 4;

            var H = Generate.Ones(n + 2, n + 2);
            var U = Generate.Ones(n + 2, n + 2);
            var V = Generate.Ones(n + 2, n + 2);
            var Hx = Generate.Ones(n + 1, n + 1);
            var Ux = Generate.Ones(n + 1, n + 1);
            var Vx = Generate.Ones(n + 1, n + 1);
            var Hy = Generate.Ones(n + 1, n + 1);
            var Uy = Generate.Ones(n + 1, n + 1);
            var Vy = Generate.Ones(n + 1, n + 1);

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
            return Add.Reduce(Add.Reduce(H / n)).Value[0];
        }
    }
}
