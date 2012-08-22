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
        /// Actually executes a binary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IBinaryOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (in1.DataAccessor.Length == 1 && in1.DataAccessor.GetType() == typeof(DefaultAccessor<T>))
                UFunc_Op_Inner_Binary_LhsScalar_Flush(op, in1.DataAccessor[0], in2, @out);
            else if (in2.DataAccessor.Length == 1 && in2.DataAccessor.GetType() == typeof(DefaultAccessor<T>))
                UFunc_Op_Inner_Binary_RhsScalar_Flush(op, in1, in2.DataAccessor[0], @out);
            else
            {
                if (UnsafeAPI.UFunc_Op_Inner_Binary_Flush_Unsafe(op, in1, in2, ref @out))
                    return;

                UFunc_Op_Inner_BinaryConv_Flush<T, T, C>(op, in1, in2, @out);
            }
        }


        /// <summary>
        /// Actually executes a binary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IBinaryOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="scalar">The left-hand-side scalar argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_LhsScalar_Flush<T, C>(C op, T scalar, NdArray<T> in2, NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (UnsafeAPI.UFunc_Op_Inner_Binary_LhsScalar_Flush_Unsafe(op, scalar, in2, ref @out))
                return;

            UFunc_Op_Inner_BinaryConv_LhsScalar_Flush<T, T, C>(op, scalar, in2, @out);
        }

		/// <summary>
        /// Actually executes a binary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IBinaryOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="scalar">The right-hand-side scalar argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_RhsScalar_Flush<T, C>(C op, NdArray<T> in1, T scalar, NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (UnsafeAPI.UFunc_Op_Inner_Binary_RhsScalar_Flush_Unsafe(op, in1, scalar, ref @out))
                return;

            UFunc_Op_Inner_BinaryConv_RhsScalar_Flush<T, T, C>(op, in1, scalar, @out);
        }

		/// <summary>
        /// The inner execution of a <see cref="T:NumCIL.IUnaryOp{0}"/>.
        /// This will just call the UnaryConv flush operation with Ta and Tb set to T
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush<T, C>(C op, NdArray<T> in1, NdArray<T> @out)
            where C : struct, IUnaryOp<T>
        {
            if (in1.DataAccessor.Length == 1 && in1.DataAccessor.GetType() == typeof(DefaultAccessor<T>))
                UFunc_Op_Inner_Unary_Flush_Scalar<T, C>(op, in1.DataAccessor[0], @out);
            else
            {
                if (!UnsafeAPI.UFunc_Op_Inner_Unary_Flush_Unsafe<T, C>(op, in1, ref @out))
                    UFunc_Op_Inner_UnaryConv_Flush<T, T, C>(op, in1, @out);
            }
        }

        /// <summary>
        /// The inner execution of a <see cref="T:NumCIL.IUnaryConvOp{0}"/>.
        /// This method will always call the unary conv flush method, because the lazy evaluation system does not implement support for handling conversion operations yet.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_UnaryConv<Ta, Tb, C>(NdArray<Ta> in1, NdArray<Tb> @out)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            UFunc_Op_Inner_UnaryConv_Flush<Ta, Tb, C>(new C(), in1, @out);
        }

        /// <summary>
        /// Actually executes a unary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IUnaryOp{0}"/> or <see cref="T:NumCIL.IUnaryConvOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_UnaryConv_Flush<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out)
            where C : IUnaryConvOp<Ta, Tb>
        {
            Ta[] d1 = in1.AsArray();
            Tb[] d2 = @out.AsArray();

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

        /// <summary>
        /// The inner execution of a <see cref="T:NumCIL.IUnaryConvOp{0}"/>.
        /// This method will always call the unary conv flush method, because the lazy evaluation system does not implement support for handling conversion operations yet.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="scalar">The input scalar</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_UnaryConv_Scalar<Ta, Tb, C>(Ta scalar, NdArray<Tb> @out)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            UFunc_Op_Inner_UnaryConv_Flush_Scalar<Ta, Tb, C>(new C(), scalar, @out);
        }

		/// <summary>
        /// The inner execution of a <see cref="T:NumCIL.IUnaryOp{0}"/>.
        /// This will just call the UnaryConv flush operation with Ta and Tb set to T
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="scalar">The input scalar</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_Scalar<T, C>(C op, T scalar, NdArray<T> @out)
            where C : struct, IUnaryOp<T>
        {
            if (!UnsafeAPI.UFunc_Op_Inner_Unary_Scalar_Flush_Unsafe<T, C>(op, scalar, ref @out))
                UFunc_Op_Inner_UnaryConv_Flush_Scalar<T, T, C>(op, scalar, @out);
        }

		/// <summary>
        /// Actually executes a unary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IUnaryOp{0}"/> or <see cref="T:NumCIL.IUnaryConvOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="scalar">The input scalar</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_UnaryConv_Flush_Scalar<Ta, Tb, C>(C op, Ta scalar, NdArray<Tb> @out)
            where C : IUnaryConvOp<Ta, Tb>
        {
            Tb[] d2 = @out.AsArray();

            if (@out.Shape.Dimensions.Length == 1)
            {
                long totalOps = @out.Shape.Dimensions[0].Length;
                long ix2 = @out.Shape.Offset;
                long stride2 = @out.Shape.Dimensions[0].Stride;

                for (long i = 0; i < totalOps; i++)
                {
                    d2[ix2] = op.Op(scalar);
                    ix2 += stride2;
                }
            }
            else if (@out.Shape.Dimensions.Length == 2)
            {
                long opsOuter = @out.Shape.Dimensions[0].Length;
                long opsInner = @out.Shape.Dimensions[1].Length;

                long ix2 = @out.Shape.Offset;
                long outerStride2 = @out.Shape.Dimensions[0].Stride;
                long innerStride2 = @out.Shape.Dimensions[1].Stride;

                outerStride2 -= innerStride2 * @out.Shape.Dimensions[1].Length;

                for (long i = 0; i < opsOuter; i++)
                {
                    for (long j = 0; j < opsInner; j++)
                    {
                        d2[ix2] = op.Op(scalar);
                        ix2 += innerStride2;
                    }

                    ix2 += outerStride2;
                }
            }
            else
            {
                long n = @out.Shape.Dimensions.LongLength - 3;
                long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                long[] counters = new long[limits.LongLength];

                long totalOps = limits.LongLength == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                //This chunck of variables are used to prevent repeated calculations of offsets
                long dimIndex0 = 0 + limits.LongLength;
                long dimIndex1 = 1 + limits.LongLength;
                long dimIndex2 = 2 + limits.LongLength;

                long opsOuter = @out.Shape.Dimensions[0 + limits.LongLength].Length;
                long opsInner = @out.Shape.Dimensions[1 + limits.LongLength].Length;
                long opsInnerInner = @out.Shape.Dimensions[2 + limits.LongLength].Length;

                long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;
                long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;
                long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;
                innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                for (long outer = 0; outer < totalOps; outer++)
                {
                    //Get the array offset for the first element in the outer dimension
                    long ix3 = @out.Shape[counters];

                    for (long i = 0; i < opsOuter; i++)
                    {
                        for (long j = 0; j < opsInner; j++)
                        {
                            for (long k = 0; k < opsInnerInner; k++)
                            {
                                d2[ix3] = op.Op(scalar);
                                ix3 += innerInnerStride3;
                            }

                            ix3 += innerStride3;
                        }

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

        /// <summary>
        /// Actually executes a nullary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.INullaryOp{0}"/> or <see cref="T:NumCIL.IUnaryConvOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="T">The type of data to generat</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary_Flush<T, C>(C op, NdArray<T> @out)
            where C : struct, INullaryOp<T>
        {
			if (UnsafeAPI.UFunc_Op_Inner_Nullary_Flush_Unsafe<T, C>(op, @out))
                return;

            T[] d = @out.AsArray();

            if (@out.Shape.Dimensions.Length == 1)
            {
                long totalOps = @out.Shape.Dimensions[0].Length;
                long ix = @out.Shape.Offset;
                long stride = @out.Shape.Dimensions[0].Stride;

                for (long i = 0; i < totalOps; i++)
                {
                    d[ix] = op.Op();
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

        /// <summary>
        /// Actually executes a binary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IBinaryOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_BinaryConv_Flush<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out)
            where C : struct, IBinaryConvOp<Ta, Tb>
        {
            if (in1.DataAccessor.Length == 1 && in1.DataAccessor.GetType() == typeof(DefaultAccessor<Ta>))
            {
                UFunc_Op_Inner_BinaryConv_LhsScalar_Flush<Ta, Tb, C>(op, in1.DataAccessor[0], in2, @out);
                return;
            }
            else if (in2.DataAccessor.Length == 1 && in2.DataAccessor.GetType() == typeof(DefaultAccessor<Ta>))
            {
                UFunc_Op_Inner_BinaryConv_RhsScalar_Flush<Ta, Tb, C>(op, in1, in2.DataAccessor[0], @out);
                return;
            }

            Ta[] d1 = in1.AsArray();
            Ta[] d2 = in2.AsArray();
            Tb[] d3 = @out.AsArray();

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


        /// <summary>
        /// Actually executes a binary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IBinaryOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="scalar">The left-hand-side scalar argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_BinaryConv_LhsScalar_Flush<Ta, Tb, C>(C op, Ta scalar, NdArray<Ta> in2, NdArray<Tb> @out)
            where C : struct, IBinaryConvOp<Ta, Tb>
        {
            Ta[] d2 = in2.AsArray();
            Tb[] d3 = @out.AsArray();

            if (@out.Shape.Dimensions.Length == 1)
            {
                long totalOps = @out.Shape.Dimensions[0].Length;

                long ix2 = in2.Shape.Offset;
                long ix3 = @out.Shape.Offset;

                long stride2 = in2.Shape.Dimensions[0].Stride;
                long stride3 = @out.Shape.Dimensions[0].Stride;

                if (stride2 == stride3 && ix2 == ix3)
                {
                    //Best case, all are equal, just keep a single counter
                    for (long i = 0; i < totalOps; i++)
                    {
                        d3[ix2] = op.Op(scalar, d2[ix2]);
                        ix2 += stride2;
                    }
                }
                else
                {
                    for (long i = 0; i < totalOps; i++)
                    {
                        //We need all three counters
                        d3[ix3] = op.Op(scalar, d2[ix2]);
                        ix2 += stride2;
                        ix3 += stride3;
                    }
                }
            }
            else if (@out.Shape.Dimensions.Length == 2)
            {
                long opsOuter = @out.Shape.Dimensions[0].Length;
                long opsInner = @out.Shape.Dimensions[1].Length;

                long ix2 = in2.Shape.Offset;
                long ix3 = @out.Shape.Offset;

                long outerStride2 = in2.Shape.Dimensions[0].Stride;
                long outerStride3 = @out.Shape.Dimensions[0].Stride;

                long innerStride2 = in2.Shape.Dimensions[1].Stride;
                long innerStride3 = @out.Shape.Dimensions[1].Stride;

                outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;
                outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                //Loop unrolling here gives a marginal speed increase

                long remainder = opsInner % 4;
                long fulls = opsInner / 4;

                for (long i = 0; i < opsOuter; i++)
                {
                    for (long j = 0; j < fulls; j++)
                    {
                        d3[ix3] = op.Op(scalar, d2[ix2]);
                        ix2 += innerStride2;
                        ix3 += innerStride3;
                        d3[ix3] = op.Op(scalar, d2[ix2]);
                        ix2 += innerStride2;
                        ix3 += innerStride3;
                        d3[ix3] = op.Op(scalar, d2[ix2]);
                        ix2 += innerStride2;
                        ix3 += innerStride3;
                        d3[ix3] = op.Op(scalar, d2[ix2]);
                        ix2 += innerStride2;
                        ix3 += innerStride3;
                    }

                    switch (remainder)
                    {
                        case 1:
                            d3[ix3] = op.Op(scalar, d2[ix2]);
                            ix2 += innerStride2;
                            ix3 += innerStride3;
                            break;
                        case 2:
                            d3[ix3] = op.Op(scalar, d2[ix2]);
                            ix2 += innerStride2;
                            ix3 += innerStride3;
                            d3[ix3] = op.Op(scalar, d2[ix2]);
                            ix2 += innerStride2;
                            ix3 += innerStride3;
                            break;
                        case 3:
                            d3[ix3] = op.Op(scalar, d2[ix2]);
                            ix2 += innerStride2;
                            ix3 += innerStride3;
                            d3[ix3] = op.Op(scalar, d2[ix2]);
                            ix2 += innerStride2;
                            ix3 += innerStride3;
                            d3[ix3] = op.Op(scalar, d2[ix2]);
                            ix2 += innerStride2;
                            ix3 += innerStride3;
                            break;
                    }

                    ix2 += outerStride2;
                    ix3 += outerStride3;
                }
            }
            else
            {
                //The inner 3 dimensions are optimized
                long n = @out.Shape.Dimensions.LongLength - 3;
                long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
                long[] counters = new long[limits.LongLength];

                long totalOps = limits.Length == 0 ? 1 : limits.Aggregate<long>((a, b) => a * b);

                //This chunk of variables prevents repeated calculations of offsets
                long dimIndex0 = 0 + limits.LongLength;
                long dimIndex1 = 1 + limits.LongLength;
                long dimIndex2 = 2 + limits.LongLength;

                long opsOuter = @out.Shape.Dimensions[dimIndex0].Length;
                long opsInner = @out.Shape.Dimensions[dimIndex1].Length;
                long opsInnerInner = @out.Shape.Dimensions[dimIndex2].Length;

                long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                long outerStride3 = @out.Shape.Dimensions[dimIndex0].Stride;

                long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                long innerStride3 = @out.Shape.Dimensions[dimIndex1].Stride;

                long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;
                long innerInnerStride3 = @out.Shape.Dimensions[dimIndex2].Stride;

                outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                outerStride3 -= innerStride3 * @out.Shape.Dimensions[dimIndex1].Length;

                innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;
                innerStride3 -= innerInnerStride3 * @out.Shape.Dimensions[dimIndex2].Length;

                for (long outer = 0; outer < totalOps; outer++)
                {
                    //Get the array offset for the first element in the outer dimension
                    long ix2 = in2.Shape[counters];
                    long ix3 = @out.Shape[counters];

                    for (long i = 0; i < opsOuter; i++)
                    {
                        for (long j = 0; j < opsInner; j++)
                        {
                            for (long k = 0; k < opsInnerInner; k++)
                            {
                                d3[ix3] = op.Op(scalar, d2[ix2]);
                                ix2 += innerInnerStride2;
                                ix3 += innerInnerStride3;
                            }

                            ix2 += innerStride2;
                            ix3 += innerStride3;
                        }

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

        /// <summary>
        /// Actually executes a binary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IBinaryOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="scalar">The right-hand-side scalar argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_BinaryConv_RhsScalar_Flush<Ta, Tb, C>(C op, NdArray<Ta> in1, Ta scalar, NdArray<Tb> @out)
            where C : struct, IBinaryConvOp<Ta, Tb>
        {
            Ta[] d1 = in1.AsArray();
            Tb[] d3 = @out.AsArray();

            if (@out.Shape.Dimensions.Length == 1)
            {
                long totalOps = @out.Shape.Dimensions[0].Length;

                long ix1 = in1.Shape.Offset;
                long ix3 = @out.Shape.Offset;

                long stride1 = in1.Shape.Dimensions[0].Stride;
                long stride3 = @out.Shape.Dimensions[0].Stride;

                if (stride1 == stride3 && ix1 == ix3)
                {
                    //Best case, all are equal, just keep a single counter
                    for (long i = 0; i < totalOps; i++)
                    {
                        d3[ix1] = op.Op(d1[ix1], scalar);
                        ix1 += stride1;
                    }
                }
                else
                {
                    for (long i = 0; i < totalOps; i++)
                    {
                        //We need all three counters
                        d3[ix3] = op.Op(d1[ix1], scalar);
                        ix1 += stride1;
                        ix3 += stride3;
                    }
                }
            }
            else if (@out.Shape.Dimensions.Length == 2)
            {
                long opsOuter = @out.Shape.Dimensions[0].Length;
                long opsInner = @out.Shape.Dimensions[1].Length;

                long ix1 = in1.Shape.Offset;
                long ix3 = @out.Shape.Offset;

                long outerStride1 = in1.Shape.Dimensions[0].Stride;
                long outerStride3 = @out.Shape.Dimensions[0].Stride;

                long innerStride1 = in1.Shape.Dimensions[1].Stride;
                long innerStride3 = @out.Shape.Dimensions[1].Stride;

                outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

                //Loop unrolling here gives a marginal speed increase

                long remainder = opsInner % 4;
                long fulls = opsInner / 4;

                for (long i = 0; i < opsOuter; i++)
                {
                    for (long j = 0; j < fulls; j++)
                    {
                        d3[ix3] = op.Op(d1[ix1], scalar);
                        ix1 += innerStride1;
                        ix3 += innerStride3;
                        d3[ix3] = op.Op(d1[ix1], scalar);
                        ix1 += innerStride1;
                        ix3 += innerStride3;
                        d3[ix3] = op.Op(d1[ix1], scalar);
                        ix1 += innerStride1;
                        ix3 += innerStride3;
                        d3[ix3] = op.Op(d1[ix1], scalar);
                        ix1 += innerStride1;
                        ix3 += innerStride3;
                    }

                    switch (remainder)
                    {
                        case 1:
                            d3[ix3] = op.Op(d1[ix1], scalar);
                            ix1 += innerStride1;
                            ix3 += innerStride3;
                            break;
                        case 2:
                            d3[ix3] = op.Op(d1[ix1], scalar);
                            ix1 += innerStride1;
                            ix3 += innerStride3;
                            d3[ix3] = op.Op(d1[ix1], scalar);
                            ix1 += innerStride1;
                            ix3 += innerStride3;
                            break;
                        case 3:
                            d3[ix3] = op.Op(d1[ix1], scalar);
                            ix1 += innerStride1;
                            ix3 += innerStride3;
                            d3[ix3] = op.Op(d1[ix1], scalar);
                            ix1 += innerStride1;
                            ix3 += innerStride3;
                            d3[ix3] = op.Op(d1[ix1], scalar);
                            ix1 += innerStride1;
                            ix3 += innerStride3;
                            break;
                    }

                    ix1 += outerStride1;
                    ix3 += outerStride3;
                }
            }
            else
            {
                //The inner 3 dimensions are optimized
                long n = @out.Shape.Dimensions.LongLength - 3;
                long[] limits = @out.Shape.Dimensions.Where(x => n-- > 0).Select(x => x.Length).ToArray();
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
                                d3[ix3] = op.Op(d1[ix1], scalar);
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
