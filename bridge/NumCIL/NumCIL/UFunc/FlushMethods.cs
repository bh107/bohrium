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
        /// This class is the entry point for all methods that
        /// execute without using the lazy mechanism, meaning that
        /// they will all request lazy data (if required) and execute
        /// in the CIL/VES. Threads and unsafe methods may still be
        /// used.
        /// </summary>
        public class FlushMethods
        {
            /// <summary>
            /// Applies a binary operation using the two operands without lazy evaluation.
            /// Assumes that the target array is allocated and shaped for broadcast.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="in1">One input operand</param>
            /// <param name="in2">Another input operand</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            public static void ApplyBinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
                    where C : struct, IBinaryOp<T>
            {
                Threads.BinaryOp<T, C>(op, in1, in2, @out);
            }

            /// <summary>
            /// Applies a binary operation using the two operands without lazy evaluation.
            /// Assumes that the target array is allocated and shaped for broadcast.
            /// </summary>
            /// <typeparam name="Ta">The type of input data to operate on</typeparam>
            /// <typeparam name="Tb">The type of output data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="in1">One input operand</param>
            /// <param name="in2">Another input operand</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            public static void ApplyBinaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out)
                    where C : struct, IBinaryConvOp<Ta, Tb>
            {
                Threads.BinaryConvOp<Ta, Tb, C>(op, in1, in2, @out);
            }

            /// <summary>
            /// Applies a unary operation using the input operand without lazy evaluation.
            /// Assumes that the target array is allocated and shaped for broadcast.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="in1">The input operand</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            public static void ApplyUnaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> @out)
                where C : struct, IUnaryOp<T>
            {
                Threads.UnaryOp<T, C>(op, in1, @out);
            }

            /// <summary>
            /// Applies a unary conversion operation using the input operand without lazy evaluation.
            /// Assumes that the target array is allocated and shaped for broadcast.
            /// </summary>
            /// <typeparam name="Ta">The type of data to convert from</typeparam>
            /// <typeparam name="Tb">The type of data to convert to</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="in1">The input operand</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            public static void ApplyUnaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out)
                where C : struct, IUnaryConvOp<Ta, Tb>
            {
                Threads.UnaryConvOp<Ta, Tb, C>(op, in1, @out);
            }

            /// <summary>
            /// Applies a nullary operation to each element in the output operand without lazy evaluation.
            /// Assumes that the target array is allocated.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            public static void ApplyNullaryOp<T, C>(C op, NdArray<T> @out)
                where C : struct, INullaryOp<T>
            {
                Threads.NullaryOp<T, C>(op, @out);
            }

            /// <summary>
            /// Reduces the input argument on the specified axis without lzy evaluation.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to reduce with</typeparam>
            /// <param name="op">The operation to perform</param>
            /// <param name="in1">The input argument</param>
            /// <param name="axis">The axis to reduce</param>
            /// <param name="out">The output target</param>
            /// <returns>The output target</returns>
            public static void Reduce<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out)
                where C : struct, IBinaryOp<T>
            {
                Threads.Reduce<T, C>(op, axis, in1, @out);
            }

            /// <summary>
            /// Performs matrix multiplication on the two operands, using the supplied methods,
            /// without using lazy evaluation.
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
            public static void Matmul<T, CADD, CMUL>(CADD addop, CMUL mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
                where CADD : struct, IBinaryOp<T>
                where CMUL : struct, IBinaryOp<T>
            {
                UFunc.UFunc_Matmul_Inner_Flush<T, CADD, CMUL>(addop, mulop, in1, in2, @out);
            }
        }
    }
}
