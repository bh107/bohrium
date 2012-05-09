using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL.Unsafe
{
    internal static class Apply
    {
        /// <summary>
        /// Unsafe implementation of applying a floating point binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_Single<C>(C op, NdArray<float> in1, NdArray<float> in2, ref NdArray<float> @out)
            where C : IBinaryOp<float>
        {
            unsafe
            {
                fixed (float* d1 = in1.Data)
                fixed (float* d2 = in2.Data)
                fixed (float* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying a floating point binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_Double<C>(C op, NdArray<double> in1, NdArray<double> in2, ref NdArray<double> @out)
            where C : IBinaryOp<double>
        {
            unsafe
            {
                fixed (double* d1 = in1.Data)
                fixed (double* d2 = in2.Data)
                fixed (double* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_Int64<C>(C op, NdArray<long> in1, NdArray<long> in2, ref NdArray<long> @out)
            where C : IBinaryOp<long>
        {
            unsafe
            {
                fixed (long* d1 = in1.Data)
                fixed (long* d2 = in2.Data)
                fixed (long* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_UInt64<C>(C op, NdArray<ulong> in1, NdArray<ulong> in2, ref NdArray<ulong> @out)
            where C : IBinaryOp<ulong>
        {
            unsafe
            {
                fixed (ulong* d1 = in1.Data)
                fixed (ulong* d2 = in2.Data)
                fixed (ulong* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_Int32<C>(C op, NdArray<int> in1, NdArray<int> in2, ref NdArray<int> @out)
            where C : IBinaryOp<int>
        {
            unsafe
            {
                fixed (int* d1 = in1.Data)
                fixed (int* d2 = in2.Data)
                fixed (int* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_UInt32<C>(C op, NdArray<uint> in1, NdArray<uint> in2, ref NdArray<uint> @out)
            where C : IBinaryOp<uint>
        {
            unsafe
            {
                fixed (uint* d1 = in1.Data)
                fixed (uint* d2 = in2.Data)
                fixed (uint* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_Int16<C>(C op, NdArray<short> in1, NdArray<short> in2, ref NdArray<short> @out)
            where C : IBinaryOp<short>
        {
            unsafe
            {
                fixed (short* d1 = in1.Data)
                fixed (short* d2 = in2.Data)
                fixed (short* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_UInt16<C>(C op, NdArray<ushort> in1, NdArray<ushort> in2, ref NdArray<ushort> @out)
            where C : IBinaryOp<ushort>
        {
            unsafe
            {
                fixed (ushort* d1 = in1.Data)
                fixed (ushort* d2 = in2.Data)
                fixed (ushort* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_Byte<C>(C op, NdArray<byte> in1, NdArray<byte> in2, ref NdArray<byte> @out)
            where C : IBinaryOp<byte>
        {
            unsafe
            {
                fixed (byte* d1 = in1.Data)
                fixed (byte* d2 = in2.Data)
                fixed (byte* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer binary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush_SByte<C>(C op, NdArray<sbyte> in1, NdArray<sbyte> in2, ref NdArray<sbyte> @out)
            where C : IBinaryOp<sbyte>
        {
            unsafe
            {
                fixed (sbyte* d1 = in1.Data)
                fixed (sbyte* d2 = in2.Data)
                fixed (sbyte* d3 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = in2.Shape.Dimensions[0].Stride;
                        long stride3 = @out.Shape.Dimensions[0].Stride;

                        if (stride1 == stride2 && stride2 == stride3 && ix1 == ix2 && ix2 == ix3)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d3[ix1] = op.Op(d1[ix1], d2[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                //We need all three counters
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += stride1;
                                ix2 += stride2;
                                ix3 += stride3;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = in2.Shape.Offset;
                        long ix3 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                        //Loop unrolling here gives a marginal speed increase

                        long remainder = opsInner % 4;
                        long fulls = opsInner / 4;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < fulls; j++)
                            {
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                                d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                                ix3 += innerStride3;
                            }

                            switch (remainder)
                            {
                                case 1:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 2:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                                case 3:
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                    break;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
                            ix3 += outerStride3;
                        }
                    }
                    else
                    {
                        //The inner 3 dimensions are optimized
                        long n = in1.Shape.Dimensions.LongLength - 3;
                        long[] limits = in1.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        //This chunk of variables prevents repeated calculations of offsets
                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix2 = in2.Shape[counters];
                            long ix3 = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d3[ix3] = op.Op(d1[ix1], d2[ix2]);
                                        ix1 += innerInnerStride1;
                                        ix2 += innerInnerStride2;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix2 += innerStride2;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix2 += outerStride2;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying a floating point unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_Single<C>(C op, NdArray<float> in1, ref NdArray<float> @out)
            where C : IUnaryOp<float>
        {
            unsafe
            {
                fixed (float* d1 = in1.Data)
                fixed (float* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying a floating point unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_Double<C>(C op, NdArray<double> in1, ref NdArray<double> @out)
            where C : IUnaryOp<double>
        {
            unsafe
            {
                fixed (double* d1 = in1.Data)
                fixed (double* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_Int64<C>(C op, NdArray<long> in1, ref NdArray<long> @out)
            where C : IUnaryOp<long>
        {
            unsafe
            {
                fixed (long* d1 = in1.Data)
                fixed (long* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_UInt64<C>(C op, NdArray<ulong> in1, ref NdArray<ulong> @out)
            where C : IUnaryOp<ulong>
        {
            unsafe
            {
                fixed (ulong* d1 = in1.Data)
                fixed (ulong* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_Int32<C>(C op, NdArray<int> in1, ref NdArray<int> @out)
            where C : IUnaryOp<int>
        {
            unsafe
            {
                fixed (int* d1 = in1.Data)
                fixed (int* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_UInt32<C>(C op, NdArray<uint> in1, ref NdArray<uint> @out)
            where C : IUnaryOp<uint>
        {
            unsafe
            {
                fixed (uint* d1 = in1.Data)
                fixed (uint* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_Int16<C>(C op, NdArray<short> in1, ref NdArray<short> @out)
            where C : IUnaryOp<short>
        {
            unsafe
            {
                fixed (short* d1 = in1.Data)
                fixed (short* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_UInt16<C>(C op, NdArray<ushort> in1, ref NdArray<ushort> @out)
            where C : IUnaryOp<ushort>
        {
            unsafe
            {
                fixed (ushort* d1 = in1.Data)
                fixed (ushort* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_Byte<C>(C op, NdArray<byte> in1, ref NdArray<byte> @out)
            where C : IUnaryOp<byte>
        {
            unsafe
            {
                fixed (byte* d1 = in1.Data)
                fixed (byte* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer unary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_SByte<C>(C op, NdArray<sbyte> in1, ref NdArray<sbyte> @out)
            where C : IUnaryOp<sbyte>
        {
            unsafe
            {
                fixed (sbyte* d1 = in1.Data)
                fixed (sbyte* d2 = @out.Data)
                {
                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long stride1 = in1.Shape.Dimensions[0].Stride;
                        long stride2 = @out.Shape.Dimensions[0].Stride;


                        if (stride1 == stride2 && ix1 == ix2)
                        {
                            //Best case, all are equal, just keep a single counter
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix1] = op.Op(d1[ix1]);
                                ix1 += stride1;
                            }
                        }
                        else
                        {
                            for (long i = 0; i < totalOps; i++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += stride1;
                                ix2 += stride2;
                            }
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix1 = in1.Shape.Offset;
                        long ix2 = @out.Shape.Offset;

                        long outerStride1 = in1.Shape.Dimensions[0].Stride;
                        long outerStride2 = @out.Shape.Dimensions[0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[1].Stride;
                        long innerStride2 = @out.Shape.Dimensions[1].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                        outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d2[ix2] = op.Op(d1[ix1]);
                                ix1 += innerStride1;
                                ix2 += innerStride2;
                            }

                            ix1 += outerStride1;
                            ix2 += outerStride2;
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

                        long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                        long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                        long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                        long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                        long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                        long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                        long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                        long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;
                        long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                        outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                        innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                        innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix1 = in1.Shape[counters];
                            long ix3 = @out.Shape[counters];


                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d2[ix3] = op.Op(d1[ix1]);
                                        ix1 += innerInnerStride1;
                                        ix3 += innerInnerStride3;
                                    }

                                    ix1 += innerStride1;
                                    ix3 += innerStride3;
                                }

                                ix1 += outerStride1;
                                ix3 += outerStride3;
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
        }

        /// <summary>
        /// Unsafe implementation of applying a floating point nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_Single<C>(C op, NdArray<float> @out)
            where C : INullaryOp<float>
        {
            unsafe
            {
                fixed (float* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying a floating point nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_Double<C>(C op, NdArray<double> @out)
            where C : INullaryOp<double>
        {
            unsafe
            {
                fixed (double* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_Int64<C>(C op, NdArray<long> @out)
            where C : INullaryOp<long>
        {
            unsafe
            {
                fixed (long* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_UInt64<C>(C op, NdArray<ulong> @out)
            where C : INullaryOp<ulong>
        {
            unsafe
            {
                fixed (ulong* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_Int32<C>(C op, NdArray<int> @out)
            where C : INullaryOp<int>
        {
            unsafe
            {
                fixed (int* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_UInt32<C>(C op, NdArray<uint> @out)
            where C : INullaryOp<uint>
        {
            unsafe
            {
                fixed (uint* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_Int16<C>(C op, NdArray<short> @out)
            where C : INullaryOp<short>
        {
            unsafe
            {
                fixed (short* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_UInt16<C>(C op, NdArray<ushort> @out)
            where C : INullaryOp<ushort>
        {
            unsafe
            {
                fixed (ushort* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_Byte<C>(C op, NdArray<byte> @out)
            where C : INullaryOp<byte>
        {
            unsafe
            {
                fixed (byte* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }

        /// <summary>
        /// Unsafe implementation of applying an integer nullary operation
        /// </summary>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush_SByte<C>(C op, NdArray<sbyte> @out)
            where C : INullaryOp<sbyte>
        {
            unsafe
            {
                fixed (sbyte* d = @out.Data)
                {

                    if (@out.Shape.Dimensions.Length == 1)
                    {
                        long totalOps = @out.Shape.Dimensions[0].Length;
                        long ix = @out.Shape.Offset;
                        long stride = @out.Shape.Dimensions[0].Stride;

                        for (long i = 0; i < totalOps; i++)
                        {
                            d[i] = op.Op();
                            ix += stride;
                        }
                    }
                    else if (@out.Shape.Dimensions.Length == 2)
                    {
                        long opsOuter = @out.Shape.Dimensions[0].Length;
                        long opsInner = @out.Shape.Dimensions[1].Length;

                        long ix = @out.Shape.Offset;
                        long outerStride = @out.Shape.Dimensions[0].Stride;
                        long innerStride = @out.Shape.Dimensions[1].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[1].Length;

                        for (long i = 0; i < opsOuter; i++)
                        {
                            for (long j = 0; j < opsInner; j++)
                            {
                                d[ix] = op.Op();
                                ix += innerStride;
                            }

                            ix += outerStride;
                        }
                    }
                    else
                    {
                        long n = @out.Shape.Dimensions.LongLength - 3;
                        long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                        long[] counters = new long[limits.LongLength];

                        long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                        long dimIndex0 = 0 + limits.LongLength;
                        long dimIndex1 = 1 + limits.LongLength;
                        long dimIndex2 = 2 + limits.LongLength;

                        long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                        long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                        long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                        long outerStride = @out.Shape.Dimensions[dimIndex0].Stride;
                        long innerStride = @out.Shape.Dimensions[dimIndex1].Stride;
                        long innerInnerStride = @out.Shape.Dimensions[dimIndex2].Stride;

                        outerStride -= innerStride * @out.Shape.Dimensions[dimIndex1].Length;
                        innerStride -= innerInnerStride * @out.Shape.Dimensions[dimIndex2].Length;

                        for (long outer = 0; outer < totalOps; outer++)
                        {
                            //Get the array offset for the first element in the outer dimension
                            long ix = @out.Shape[counters];

                            for (long i = 0; i < opsOuter; i++)
                            {
                                for (long j = 0; j < opsInner; j++)
                                {
                                    for (long k = 0; k < opsInnerInner; k++)
                                    {
                                        d[ix] = op.Op();
                                        ix += innerInnerStride;
                                    }

                                    ix += innerStride;
                                }

                                ix += outerStride;
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
        }
    }
}
