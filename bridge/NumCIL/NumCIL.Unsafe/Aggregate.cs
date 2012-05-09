using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL.Unsafe
{
    internal static class Aggregate
    {
        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static float Aggregate_Entry_Single<C>(C op, NdArray<float> in1)
            where C : struct, IBinaryOp<float>
        {
            float result;
            unsafe
            {
                fixed (float* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static double Aggregate_Entry_Double<C>(C op, NdArray<double> in1)
            where C : struct, IBinaryOp<double>
        {
            double result;
            unsafe
            {
                fixed (double* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static long Aggregate_Entry_Int64<C>(C op, NdArray<long> in1)
            where C : struct, IBinaryOp<long>
        {
            long result;
            unsafe
            {
                fixed (long* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static ulong Aggregate_Entry_UInt64<C>(C op, NdArray<ulong> in1)
            where C : struct, IBinaryOp<ulong>
        {
            ulong result;
            unsafe
            {
                fixed (ulong* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static int Aggregate_Entry_Int32<C>(C op, NdArray<int> in1)
            where C : struct, IBinaryOp<int>
        {
            int result;
            unsafe
            {
                fixed (int* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static uint Aggregate_Entry_UInt32<C>(C op, NdArray<uint> in1)
            where C : struct, IBinaryOp<uint>
        {
            uint result;
            unsafe
            {
                fixed (uint* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static short Aggregate_Entry_Int16<C>(C op, NdArray<short> in1)
            where C : struct, IBinaryOp<short>
        {
            short result;
            unsafe
            {
                fixed (short* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static ushort Aggregate_Entry_UInt16<C>(C op, NdArray<ushort> in1)
            where C : struct, IBinaryOp<ushort>
        {
            ushort result;
            unsafe
            {
                fixed (ushort* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static byte Aggregate_Entry_Byte<C>(C op, NdArray<byte> in1)
            where C : struct, IBinaryOp<byte>
        {
            byte result;
            unsafe
            {
                fixed (byte* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static sbyte Aggregate_Entry_SByte<C>(C op, NdArray<sbyte> in1)
            where C : struct, IBinaryOp<sbyte>
        {
            sbyte result;
            unsafe
            {
                fixed (sbyte* d1 = in1.Data)
                {

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
                }
            }

            return result;
        }
    }
}
