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

            if (@out.m_data is ILazyAccessor<T>)
                ((ILazyAccessor<T>)@out.m_data).AddOperation(new LazyMatmulOperation<T>(addop, mulop), @out, in1, in2);
            else
                UFunc_Matmul_Inner_Flush<T, CADD, CMUL>(addop, mulop, in1, in2, @out);

            return @out;
        }

        /// <summary>
        /// Actual implmentation of the matrix multiplication operation.
        /// This implementation is not working in-place, but rather creates
        /// a combined temporary matrix of the multiplication results,
        /// which is then reduced with the add operation to produce the
        /// final result
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
        private static void UFunc_Matmul_Inner_Flush<T, CADD, CMUL>(CADD addop, CMUL mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
            where CADD : struct, IBinaryOp<T>
            where CMUL : struct, IBinaryOp<T>
        {
            //The matrix multiplication is implemented by extending
            // one operand to allow broacast compatible shapes,
            // then applying the multiplication to this enlarged array,
            // and finally reducing the result to a suitable shape
            // 
            //The operations are done using internal functions to avoid 
            // lazy evaluating the operations

            var lv = in1.Subview(Range.NewAxis, 2);

            Tuple<Shape, Shape, NdArray<T>> reshaped = SetupApplyHelper<T>(lv, in2, null);

            lv = lv.Reshape(reshaped.Item1);
            var rv = in2.Reshape(reshaped.Item2);
            var tmp = reshaped.Item3;

            UFunc_Op_Inner_Binary_Flush<T, CMUL>(mulop, lv, rv, ref tmp);

            var nx = SetupReduceHelper<T>(tmp, 1, @out);
            UFunc_Reduce_Inner_Flush<T, CADD>(addop, 1, tmp, nx);
        }
    }
}
