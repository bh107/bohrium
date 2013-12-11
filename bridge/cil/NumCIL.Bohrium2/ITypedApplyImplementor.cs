using System;
using NumCIL.Generic;

namespace NumCIL.Bohrium2
{
    internal interface ITypedApplyImplementor<T>
    {
        /// <summary>
        /// Applies a binary operation using the two operands.
        /// Assumes that the target array is allocated and shaped for broadcast.
        /// </summary>
        /// <param name="op">The operation to use</param>
        /// <param name="in1">One input operand</param>
        /// <param name="in2">Another input operand</param>
        /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
        /// <returns>True if the operation was applied, false otherwise</returns>
        bool ApplyBinaryOp(Type c, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out);
        
        /// <summary>
        /// Applies a binary operation using the two operands.
        /// Assumes that the target array is allocated and shaped for broadcast.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <param name="op">The operation to use</param>
        /// <param name="in1">One input operand</param>
        /// <param name="in2">Another input operand</param>
        /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
        /// <returns>True if the operation was applied, false otherwise</returns>
        bool ApplyBinaryConvOp<Ta>(Type c, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<T> @out);

        /// <summary>
        /// Applies a unary operation using the input operand.
        /// Assumes that the target array is allocated and shaped for broadcast.
        /// </summary>
        /// <param name="op">The operation to use</param>
        /// <param name="in1">The input operand</param>
        /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
        /// <returns>True if the operation was applied, false otherwise</returns>
        bool ApplyUnaryOp(Type c, NdArray<T> in1, NdArray<T> @out);
        
        /// <summary>
        /// Applies a unary conversion operation using the input operand.
        /// Assumes that the target array is allocated and shaped for broadcast.
        /// </summary>
        /// <typeparam name="Ta">The type of data to convert from</typeparam>
        /// <param name="op">The operation to use</param>
        /// <param name="in1">The input operand</param>
        /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
        /// <returns>True if the operation was applied, false otherwise</returns>
        bool ApplyUnaryConvOp<Ta>(Type c, NdArray<Ta> in1, NdArray<T> @out);
        
        /// <summary>
        /// Applies a nullary operation to each element in the output operand.
        /// Assumes that the target array is allocated.
        /// </summary>
        /// <param name="op">The operation to use</param>
        /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
        /// <returns>True if the operation was applied, false otherwise</returns>
        bool ApplyNullaryOp(Type c, NdArray<T> @out);
        
        /// <summary>
        /// Reduces the input argument on the specified axis.
        /// </summary>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>True if the operation was applied, false otherwise</returns>
        bool ApplyReduce(Type c, long axis, NdArray<T> in1, NdArray<T> @out);
        
        /// <summary>
        /// Performs matrix multiplication on the two operands, using the supplied methods.
        /// </summary>
        /// <param name="addop">The add operator</param>
        /// <param name="mulop">The multiply operator</param>
        /// <param name="in1">The left-hand-side argument</param>
        /// <param name="in2">The right-hand-side argument</param>
        /// <param name="out">An optional output argument, use for in-place operations</param>
        /// <returns>True if the operation was applied, false otherwise</returns>
        bool ApplyMatmul(Type cadd, Type cmul, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null);
        
        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <param name="op">The operation to reduce with</param>
        /// <param name="in1">The array to aggregate</param>
        /// <param name="result">A scalar value that is the result of aggregating all elements</param>
        /// <returns>True if the operation was applied, false otherwise</returns>            
        bool ApplyAggregate(Type c, NdArray<T> in1, out T result);
    }
}
