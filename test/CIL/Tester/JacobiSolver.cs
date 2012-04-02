using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Float;
using NumCIL;

namespace Tester
{
    using T = System.Single;
    using R = NumCIL.Range;

    public static class JacobiSolver
    {
        public static long Solve(long width, long height, long? fixedIterations = null)
        {
            var full = Generate.Zeroes(height + 2, width + 2);
            var work = Generate.Zeroes(height, width);
            var diff = Generate.Zeroes(height, width);
            var tmpdelta = Generate.Zeroes(height);

            full.Name = "full";
            work.Name = "work";
            diff.Name = "diff";
            tmpdelta.Name = "tmpdelta";

            var cells = full[R.Slice(1, -1),        R.Slice(1, -1)       ];
            var up =    full[R.Slice(1, -1),        R.Slice(0, -2)       ];
            var left =  full[R.Slice(0, -2),        R.Slice(1, -1)       ];
            var right = full[R.Slice(2, width + 1), R.Slice(1, -1)       ];
            var down =  full[R.Slice(1, -1),        R.Slice(2, width + 1)];

            full[R.All, R.El(0)] += -273.5f;
            full[R.All, R.El(-1)] += -273.5f;
            full[0] += 40f;
            full[-1] += -273.5f;

            T epsilon = width * height * 0.002f;
            T delta = epsilon + 1;
            
            int i = 0;

            while (fixedIterations.HasValue ? (i < fixedIterations.Value) : epsilon < delta)
            {
                //If we are on GPU, this will make a kernel for a loop
                //NumCIL.cphVB.Utility.Flush();

                i++;
                work[R.All] = cells;
                work += up;
                work += left;
                work += right;
                work += down;
                work *= 0.2f;

                if (!fixedIterations.HasValue)
                {
                    UFunc.Apply<T, Sub>(cells, work, diff);
                    UFunc.Apply<T, Abs>(diff, diff);
                    UFunc.Reduce<T, Add>(diff, 0, tmpdelta);
                    delta = tmpdelta.Reduce<Add>().Value[0];
                }
                cells[R.All] = work;
            }

            if (fixedIterations.HasValue)
            {
                //Access the data to ensure it is flushed
                var token = full.Data[0];
            }

            return i;
        }
    }
}
