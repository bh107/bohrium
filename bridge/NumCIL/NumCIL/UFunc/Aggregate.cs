#region Copyright
/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL
{
    public static partial class UFunc
    {
        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="T">The value to operate on</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        public static T Aggregate<T>(IBinaryOp<T> op, NdArray<T> in1)
        {
            var method = typeof(UFunc).GetMethod("Aggregate_Entry", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(T), op.GetType());
            return (T)gm.Invoke(null, new object[] { op, in1 });
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="T">The value to operate on</typeparam>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        public static T Aggregate<T, C>(NdArray<T> in1)
            where C : struct, IBinaryOp<T>
        {
            return Aggregate_Entry<T, C>(new C(), in1);
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="T">The value to operate on</typeparam>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static T Aggregate_Entry<T, C>(C op, NdArray<T> in1)
            where C : struct, IBinaryOp<T>
        {
            T result;
            if (UnsafeAPI.Aggregate_Entry_Unsafe<T, C>(op, in1, out result))
                return result;

            T[] d1 = in1.AsArray();

            if (in1.Shape.Dimensions.Length == 1)
            {
                long totalOps = in1.Shape.Dimensions[0].Length;
                long ix1 = in1.Shape.Offset;
                long stride1 = in1.Shape.Dimensions[0].Stride;

                result = d1[ix1];
                ix1 += stride1;

                for (long i = 1; i < totalOps; i++)
                {
                    result = op.Op(result, d1[ix1]);
                    ix1 += stride1;
                }
            }
            else if (in1.Shape.Dimensions.Length == 2)
            {
                long opsOuter = in1.Shape.Dimensions[0].Length;
                long opsInner = in1.Shape.Dimensions[1].Length;

                long ix1 = in1.Shape.Offset;
                long outerStride1 = in1.Shape.Dimensions[0].Stride;
                long innerStride1 = in1.Shape.Dimensions[1].Stride;
                outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;

                result = d1[ix1];
                ix1 += innerStride1;

                for (long i = 0; i < opsOuter; i++)
                {
                    for (long j = (i == 0 ? 1 : 0); j < opsInner; j++)
                    {
                        result = op.Op(result, d1[ix1]);
                        ix1 += innerStride1;
                    }

                    ix1 += outerStride1;
                }
            }
            else
            {
                long n = in1.Shape.Dimensions.LongLength - 3;
                long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                long[] counters = new long[limits.LongLength];

                long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                //This chunck of variables are used to prevent repeated calculations of offsets
                long dimIndex0 = 0 + limits.LongLength;
                long dimIndex1 = 1 + limits.LongLength;
                long dimIndex2 = 2 + limits.LongLength;

                long opsOuter = in1.Shape.Dimensions[0 + limits.LongLength].Length;
                long opsInner = in1.Shape.Dimensions[1 + limits.LongLength].Length;
                long opsInnerInner = in1.Shape.Dimensions[2 + limits.LongLength].Length;

                long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;

                outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;

                result = d1[in1.Shape[counters]];
                bool first = true;

                for (long outer = 0; outer < totalOps; outer++)
                {
                    //Get the array offset for the first element in the outer dimension
                    long ix1 = in1.Shape[counters];
                    if (first)
                        ix1 += innerInnerStride1;

                    for (long i = 0; i < opsOuter; i++)
                    {
                        for (long j = 0; j < opsInner; j++)
                        {
                            for (long k = (first ? 1 : 0); k < opsInnerInner; k++)
                            {
                                result = op.Op(result, d1[ix1]);
                                ix1 += innerInnerStride1;
                            }
                            first = false;

                            ix1 += innerStride1;
                        }

                        ix1 += outerStride1;
                    }

                    if (counters.LongLength > 0)
                    {
                        //Basically a ripple carry adder
                        long p = counters.LongLength - 1;
                        while (++counters[p] == limits[p] && p > 0)
                        {
                            counters[p] = 0;
                            p--;
                        }
                    }
                }
            }

            return result;
        }
    }
}
