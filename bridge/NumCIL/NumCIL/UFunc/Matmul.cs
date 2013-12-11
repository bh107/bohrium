#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
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
using NumCIL.Generic;

namespace NumCIL
{
    partial class UFunc
    {
        /// <summary>
        /// Wrapper class to represent a pending matrix multiplication operation in a list of pending operations
        /// </summary>
        /// <typeparam name="T">The type of data being processed</typeparam>
        public class LazyMatmulOperation<T>
            : IOp<T>
        {
            /// <summary>
            /// The add operator
            /// </summary>
            public readonly IBinaryOp<T> AddOperator;
            /// <summary>
            /// The multiply operator
            /// </summary>
            public readonly IBinaryOp<T> MulOperator;

            /// <summary>
            /// Initializes a new instance of the <see cref="LazyMatmulOperation&lt;T&gt;"/> class
            /// </summary>
            /// <param name="addOperation">The add operator</param>
            /// <param name="mulOperation">The multiply operator</param>
            public LazyMatmulOperation(IBinaryOp<T> addOperation, IBinaryOp<T> mulOperation)
            {
                this.AddOperator = addOperation;
                this.MulOperator = mulOperation;
            }
        }

        /// <summary>
        /// Performs matrix multiplication on the two operands, using the supplied methods
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="addop">The add operator</param>
        /// <param name="mulop">The multiply operator</param>
        /// <param name="in1">The left-hand-side argument</param>
        /// <param name="in2">The right-hand-side argument</param>
        /// <param name="out">An optional output argument, use for in-place operations</param>
        /// <returns>An array with the matrix multiplication result</returns>
        public static NdArray<T> Matmul<T>(IBinaryOp<T> addop, IBinaryOp<T> mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
        {
            var method = typeof(UFunc).GetMethod("Matmul_Entry", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(T), addop.GetType(), mulop.GetType());
            return (NdArray<T>)gm.Invoke(null, new object[] { addop, mulop, in1, in2, @out });
        }

        /// <summary>
        /// Performs matrix multiplication on the two operands, using the supplied methods
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="CADD">The typed add operator</typeparam>
        /// <typeparam name="CMUL">The typed multiply operator</typeparam>
        /// <param name="in1">The left-hand-side argument</param>
        /// <param name="in2">The right-hand-side argument</param>
        /// <param name="out">An optional output argument, use for in-place operations</param>
        /// <returns>An array with the matrix multiplication result</returns>
        public static NdArray<T> Matmul<T, CADD, CMUL>(NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
            where CADD : struct, IBinaryOp<T>
            where CMUL : struct, IBinaryOp<T>
        {
            return Matmul_Entry<T, CADD, CMUL>(new CADD(), new CMUL(), in1, in2, @out);
        }


        /// <summary>
        /// Performs matrix multiplication on the two operands, using the supplied methods.
        /// This is the main entry point for detecting valid inputs, instanciating the output,
        /// and determining if the operation should be lazy evaluated.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="CADD">The typed add operator</typeparam>
        /// <typeparam name="CMUL">The typed multiply operator</typeparam>
        /// <param name="addop">The add operator</param>
        /// <param name="mulop">The multiply operator</param>
        /// <param name="in1">The left-hand-side argument</param>
        /// <param name="in2">The right-hand-side argument</param>
        /// <param name="out">An optional output argument, use for in-place operations</param>
        /// <returns>An array with the matrix multiplication result</returns>
        private static NdArray<T> Matmul_Entry<T, CADD, CMUL>(CADD addop, CMUL mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
            where CADD : struct, IBinaryOp<T>
            where CMUL : struct, IBinaryOp<T>
        {
            if (in1.Shape.Dimensions.LongLength != 2)
                throw new ArgumentException("Input elements must be 2D", "in1");
            if (in2.Shape.Dimensions.LongLength > 2)
                throw new ArgumentException("Input elements must be 2D", "in2");
            if (in1.Shape.Dimensions[1].Length != in2.Shape.Dimensions[0].Length)
                throw new ArgumentException(string.Format("Input elements shape size must match for matrix multiplication"));

            if (in2.Shape.Dimensions.LongLength < 2)
                in2 = in2.Subview(Range.NewAxis, in2.Shape.Dimensions.LongLength);

            long[] newDims = new long[] { in1.Shape.Dimensions[0].Length, in2.Shape.Dimensions[1].Length };
            if (@out == null)
                @out = new NdArray<T>(new Shape(newDims));
            else
            {
                if (@out.Shape.Dimensions.LongLength != 2 || @out.Shape.Dimensions[0].Length != newDims[0] || @out.Shape.Dimensions[1].Length != newDims[1])
                    throw new Exception("The output array for matrix multiplication is not correctly shaped");
            }

            if (@out.DataAccessor is ILazyAccessor<T>)
                ((ILazyAccessor<T>)@out.DataAccessor).AddOperation(new LazyMatmulOperation<T>(addop, mulop), @out, in1, in2);
            else
                ApplyManager.ApplyMatmul<T, CADD, CMUL>(addop, mulop, in1, in2, @out);

            return @out;
        }

        /// <summary>
        /// Actual implmentation of the matrix multiplication operation.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="CADD">The typed add operator</typeparam>
        /// <typeparam name="CMUL">The typed multiply operator</typeparam>
        /// <param name="addop">The add operator</param>
        /// <param name="mulop">The multiply operator</param>
        /// <param name="in1">The left-hand-side argument</param>
        /// <param name="in2">The right-hand-side argument</param>
        /// <param name="out">An optional output argument, use for in-place operations</param>
        private static void UFunc_Matmul_Inner_Flush<T, CADD, CMUL>(CADD addop, CMUL mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
            where CADD : struct, IBinaryOp<T>
            where CMUL : struct, IBinaryOp<T>
        {
            //Matrix multiplication of two vectors is the dot product
            if (@out.Shape.Elements == 1)
            {
                @out.Value[0] = UFunc_CombineAndAggregate_Inner_Flush<T, CADD, CMUL>(addop, mulop, in1, in2);
                return;
            }

            if (@out.Shape.Dimensions.LongLength != 2)
                throw new Exception("Matrix multiplication is only supported for 1 and 2 dimensional arrays");

            long opsOuter = @out.Shape.Dimensions[0].Length;
            long opsInner = @out.Shape.Dimensions[1].Length;
            long opsInnerInner = in1.Shape.Dimensions[1].Length;

            T[] d1 = in1.AsArray();
            T[] d2 = in2.AsArray();
            T[] d3 = @out.AsArray();

            long ix1Base = in1.Shape.Offset;
            long outerStride1 = in1.Shape.Dimensions[0].Stride;
            long innerStride1 = in1.Shape.Dimensions[1].Stride;
            //outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;

            NdArray<T> rhs = in2.Transpose();
            long ix2Base = rhs.Shape.Offset;
            long outerStride2 = rhs.Shape.Dimensions[0].Stride;
            long innerStride2 = rhs.Shape.Dimensions[1].Stride;
            outerStride2 -= innerStride2 * rhs.Shape.Dimensions[1].Length;

            long ix3 = @out.Shape.Offset;
            long outerStride3 = @out.Shape.Dimensions[0].Stride;
            long innerStride3 = @out.Shape.Dimensions[1].Stride;
            outerStride3 -= innerStride3 * @out.Shape.Dimensions[1].Length;

            for (long i = 0; i < opsOuter; i++)
            {
                long ix2 = ix2Base;

                for (long j = 0; j < opsInner; j++)
                {
                    long ix1 = ix1Base;

                    T result = mulop.Op(d1[ix1], d2[ix2]);
                    ix1 += innerStride1;
                    ix2 += innerStride2;

                    for(long k = 1; k < opsInnerInner; k++)
                    {
                        result = addop.Op(result, mulop.Op(d1[ix1], d2[ix2]));
                        ix1 += innerStride1;
                        ix2 += innerStride2;
                    }


                    d3[ix3] = result;
                    ix3 += innerStride3;
                    ix2 += outerStride2;
                }

                ix3 += outerStride3;
                ix1Base += outerStride1;
            }
        }

        /// <summary>
        /// Actual implmentation of an operation that performs combination of two NdArray values and aggregates the results.
        /// If the aggregation operation is addition, and the combination operation is multiplication, the operation is essentially the dot product, 
        /// but the operation aggregation occurs accross all dimensions.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="CAGGREGATE">The typed add operator</typeparam>
        /// <typeparam name="CCOMBINE">The typed multiply operator</typeparam>
        /// <param name="aggregate">The add operator</param>
        /// <param name="combine">The multiply operator</param>
        /// <param name="in1">The left-hand-side argument</param>
        /// <param name="in2">The right-hand-side argument</param>
        private static T UFunc_CombineAndAggregate_Inner_Flush<T, CAGGREGATE, CCOMBINE>(CAGGREGATE aggregate, CCOMBINE combine, NdArray<T> in1, NdArray<T> in2)
            where CAGGREGATE : struct, IBinaryOp<T>
            where CCOMBINE : struct, IBinaryOp<T>
        {
            T result;
            T[] d1 = in1.AsArray();
            T[] d2 = in2.AsArray();

            if (in1.Shape.Dimensions.Length == 1)
            {
                long totalOps = in1.Shape.Dimensions[0].Length;
                long ix1 = in1.Shape.Offset;
                long ix2 = in2.Shape.Offset;

                long stride1 = in1.Shape.Dimensions[0].Stride;
                long stride2 = in2.Shape.Dimensions[0].Stride;

                result = combine.Op(d1[ix1], d2[ix2]);
                ix1 += stride1;
                ix2 += stride2;

                for (long i = 1; i < totalOps; i++)
                {
                    result = aggregate.Op(result, combine.Op(d1[ix1], d2[ix2]));
                    ix1 += stride1;
                    ix2 += stride2;
                }
            }
            else if (in1.Shape.Dimensions.Length == 2)
            {
                long opsOuter = in1.Shape.Dimensions[0].Length;
                long opsInner = in1.Shape.Dimensions[1].Length;

                long ix1 = in1.Shape.Offset;
                long ix2 = in2.Shape.Offset;

                long outerStride1 = in1.Shape.Dimensions[0].Stride;
                long outerStride2 = in2.Shape.Dimensions[0].Stride;

                long innerStride1 = in1.Shape.Dimensions[1].Stride;
                long innerStride2 = in2.Shape.Dimensions[1].Stride;

                outerStride1 -= innerStride1 * in1.Shape.Dimensions[1].Length;
                outerStride2 -= innerStride2 * in2.Shape.Dimensions[1].Length;

                result = combine.Op(d1[ix1], d2[ix2]);
                ix1 += innerStride1;
                ix2 += innerStride2;

                for (long i = 0; i < opsOuter; i++)
                {
                    for (long j = (i == 0 ? 1 : 0); j < opsInner; j++)
                    {
                        result = aggregate.Op(result, combine.Op(d1[ix1], d2[ix2]));
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

                long opsOuter = in1.Shape.Dimensions[dimIndex0].Length;
                long opsInner = in1.Shape.Dimensions[dimIndex1].Length;
                long opsInnerInner = in1.Shape.Dimensions[dimIndex2].Length;

                long outerStride1 = in1.Shape.Dimensions[dimIndex0].Stride;
                long innerStride1 = in1.Shape.Dimensions[dimIndex1].Stride;
                long innerInnerStride1 = in1.Shape.Dimensions[dimIndex2].Stride;

                long outerStride2 = in2.Shape.Dimensions[dimIndex0].Stride;
                long innerStride2 = in2.Shape.Dimensions[dimIndex1].Stride;
                long innerInnerStride2 = in2.Shape.Dimensions[dimIndex2].Stride;

                outerStride1 -= innerStride1 * in1.Shape.Dimensions[dimIndex1].Length;
                innerStride1 -= innerInnerStride1 * in1.Shape.Dimensions[dimIndex2].Length;
                outerStride2 -= innerStride2 * in2.Shape.Dimensions[dimIndex1].Length;
                innerStride2 -= innerInnerStride2 * in2.Shape.Dimensions[dimIndex2].Length;

                result = combine.Op(d1[in1.Shape[counters]], d2[in2.Shape[counters]]);
                bool first = true;

                for (long outer = 0; outer < totalOps; outer++)
                {
                    //Get the array offset for the first element in the outer dimension
                    long ix1 = in1.Shape[counters];
                    long ix2 = in2.Shape[counters];
                    if (first)
                    {
                        ix1 += innerInnerStride1;
                        ix2 += innerInnerStride2;
                    }

                    for (long i = 0; i < opsOuter; i++)
                    {
                        for (long j = 0; j < opsInner; j++)
                        {
                            for (long k = (first ? 1 : 0); k < opsInnerInner; k++)
                            {
                                result = aggregate.Op(result, combine.Op(d1[ix1], d2[ix2]));
                                ix1 += innerInnerStride1;
                                ix2 += innerInnerStride2;
                            }
                            first = false;

                            ix1 += innerStride1;
                            ix2 += innerStride2;
                        }

                        ix1 += outerStride1;
                        ix2 += outerStride2;
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
