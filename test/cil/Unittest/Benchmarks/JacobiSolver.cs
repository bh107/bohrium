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
using NumCIL.Double;
using NumCIL;

namespace UnitTest.Benchmarks
{
    using T = System.Double;
    using R = NumCIL.Range;

    public static class JacobiSolver
    {
        public static T Solve(long width, long height, bool calculateDelta, long? fixedIterations = null)
        {
            if (fixedIterations == null)
                calculateDelta = true;

            var full = Generate.Zeroes(height + 2, width + 2);
            var work = Generate.Zeroes(height, width);
            var diff = Generate.Zeroes(height, width);
            var tmpdelta = Generate.Zeroes(width);
            var deltares = Generate.Empty(1);

            full.Name = "full";
            work.Name = "work";
            diff.Name = "diff";
            tmpdelta.Name = "tmpdelta";

            var cells = full[R.Slice(1, -1),  R.Slice(1, -1) ];
            var up =    full[R.Slice(1, -1),  R.Slice(0, -2) ];
            var left =  full[R.Slice(0, -2),  R.Slice(1, -1) ];
            var right = full[R.Slice(2,  0),  R.Slice(1, -1) ];
            var down =  full[R.Slice(1, -1),  R.Slice(2,  0) ];

            cells.Name = "cells";
            up.Name = "up";
            left.Name = "left";
            right.Name = "right";
            down.Name = "down";

            full[R.All, R.El(0)] += -273.5f;
            full[R.All, R.El(-1)] += -273.5f;
            full[0] += 40f;
            full[-1] += -273.5f;

            T epsilon = width * height * 0.002f;
            T delta = epsilon + 1;

            int i = 0;

            work[R.All] = cells;

            while (fixedIterations.HasValue ? (i < fixedIterations.Value) : epsilon < delta)
            {
                i++;
                Add.Apply(work, up, work);
                Add.Apply(work, left, work);
                Add.Apply(work, right, work);
                Add.Apply(work, down, work);
                Mul.Apply(work, 0.2f, work);
                
                //This will do the same but not in-place
                // We need to recompile the kernels to 
                // support this

                /*work += up;
                work += left;
                work += right;
                work += down;
                work *= 0.2f;*/

                if (calculateDelta)
                {
                    Sub.Apply(cells, work, diff);
                    Abs.Apply(diff, diff);
                    Add.Reduce(diff, 0, tmpdelta);
                    delta = Add.Reduce(tmpdelta, 0, deltares).Value[0];
                }
                cells[R.All] = work;
            }

            if (calculateDelta)
                return delta;
            else
                //Access the data to ensure it is flushed
                return full.Value[0];
        }
    }
}
