using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL
{
    /// <summary>
    /// Basic marker interface for all operations
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface IOp<T> { }

    /// <summary>
    /// Describes an operation that takes two arguments and produce an output
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface IBinaryOp<T> : IOp<T> 
    { 
        /// <summary>
        /// Performs the operation
        /// </summary>
        /// <param name="a">Left-hand-side input value</param>
        /// <param name="b">Right-hand-side input value</param>
        /// <returns>The result of applying the operation</returns>
        T Op(T a, T b); 
    }
    
    /// <summary>
    /// Describes an operation that takes an input argument and produce an ouput
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface IUnaryOp<T> : IUnaryConvOp<T, T> { };
    /// <summary>
    /// Describes an operation that takes an input argument and produce an ouput
    /// </summary>
    /// <typeparam name="Ta">The input data type</typeparam>
    /// <typeparam name="Tb">The output data type</typeparam>
    public interface IUnaryConvOp<Ta, Tb> : IOp<Ta> 
    { 
        /// <summary>
        /// Performs the operation
        /// </summary>
        /// <param name="a">The input argument</param>
        /// <returns>The converted value</returns>
        Tb Op(Ta a); 
    }

    /// <summary>
    /// Describes an operation that takes no inputs but produces an output
    /// </summary>
    /// <typeparam name="T">The type of data to produce</typeparam>
    public interface INullaryOp<T>: IOp<T> 
    {
        /// <summary>
        /// Performs an operation
        /// </summary>
        /// <returns>The result of the operation</returns>
        T Op(); 
    }

    /// <summary>
    /// An operation that outputs the same value for each input
    /// </summary>
    /// <typeparam name="T">The type of data to produce</typeparam>
    public struct GenerateOp<T> : INullaryOp<T> 
    { 
        /// <summary>
        /// The value all elements are assigned
        /// </summary>
        public readonly T Value;
        /// <summary>
        /// Constructs a new GenerateOp with the specified value
        /// </summary>
        /// <param name="value"></param>
        public GenerateOp(T value) { Value = value; }
        /// <summary>
        /// Executes the operation, i.e. returns the value
        /// </summary>
        /// <returns>The result value to assign</returns>
        public T Op() { return Value; } 
    }

    /// <summary>
    /// An operation that copies data from one element to another, aka the identity operation
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public struct CopyOp<T> : IUnaryOp<T> 
    {
        /// <summary>
        /// Returns the input value
        /// </summary>
        /// <param name="a">The value to return</param>
        /// <returns>The input value</returns>
        public T Op(T a) { return a; } 
    }

    /// <summary>
    /// An operation that is implemented with a lambda function.
    /// Note that the operation is executed as a virtual function call,
    /// and thus induces some overhead to each invocation.
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public struct BinaryLambdaOp<T> : IBinaryOp<T>
    {
        /// <summary>
        /// The local function reference
        /// </summary>
        private readonly Func<T, T, T> m_op;
        /// <summary>
        /// Constructs a BinaryLambdaOp from a lambda function
        /// </summary>
        /// <param name="op">The lambda function to wrap</param>
        public BinaryLambdaOp(Func<T, T, T> op) { m_op = op; }
        /// <summary>
        /// Executes the operation
        /// </summary>
        /// <param name="a">Input data a</param>
        /// <param name="b">Intput data b</param>
        /// <returns>The result of invoking the function</returns>
        public T Op(T a, T b) { return m_op(a, b); }
        /// <summary>
        /// Convenience method to allow using a lambda function as an operator
        /// </summary>
        /// <param name="op">The lambda function</param>
        /// <returns>A BinaryLambdaOp that wraps the function</returns>
        public static implicit operator BinaryLambdaOp<T>(Func<T, T, T> op) { return new BinaryLambdaOp<T>(op); }
    }

    /// <summary>
    /// An operation that is implemented with a lambda function.
    /// Note that the operation is executed as a virtual function call,
    /// and thus induces some overhead to each invocation.
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public struct UnaryLambdaOp<T> : IUnaryOp<T>
    {
        /// <summary>
        /// The local function reference
        /// </summary>
        private readonly Func<T, T> m_op;
        /// <summary>
        /// Constructs a UnaryLambdaOp from a lambda function
        /// </summary>
        /// <param name="op">The lambda function to wrap</param>
        public UnaryLambdaOp(Func<T, T> op) { m_op = op; }
        /// <summary>
        /// Executes the operation
        /// </summary>
        /// <param name="a">Input data</param>
        /// <returns>The result of invoking the function</returns>
        public T Op(T a) { return m_op(a); }
        /// <summary>
        /// Convenience method to allow using a lambda function as an operator
        /// </summary>
        /// <param name="op">The lambda function</param>
        /// <returns>A UnaryLambdaOp that wraps the function</returns>
        public static implicit operator UnaryLambdaOp<T>(Func<T, T> op) { return new UnaryLambdaOp<T>(op); }
    }

    /// <summary>
    /// An operation that is implemented with a lambda function.
    /// Note that the operation is executed as a virtual function call,
    /// and thus induces some overhead to each invocation.
    /// </summary>
    /// <typeparam name="Ta">The input data type</typeparam>
    /// <typeparam name="Tb">The output data type</typeparam>
    public struct UnaryConvLambdaOp<Ta, Tb> : IUnaryConvOp<Ta, Tb>
    {
        /// <summary>
        /// The local function reference
        /// </summary>
        private readonly Func<Ta, Tb> m_op;
        /// <summary>
        /// Constructs a UnaryConvLambdaOp from a lambda function
        /// </summary>
        /// <param name="op">The lambda function to wrap</param>
        public UnaryConvLambdaOp(Func<Ta, Tb> op) { m_op = op; }
        /// <summary>
        /// Executes the operation
        /// </summary>
        /// <param name="a">Input data</param>
        /// <returns>The result of invoking the function</returns>
        public Tb Op(Ta a) { return m_op(a); }
        /// <summary>
        /// Convenience method to allow using a lambda function as an operator
        /// </summary>
        /// <param name="op">The lambda function</param>
        /// <returns>A UnaryConvLambdaOp that wraps the function</returns>
        public static implicit operator UnaryConvLambdaOp<Ta, Tb>(Func<Ta, Tb> op) { return new UnaryConvLambdaOp<Ta, Tb>(op); }
    }

    /// <summary>
    /// Interface to allow reading the scalar value from a ScalarOp.
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface ScalarAccess<T>
    {
        /// <summary>
        /// The operation applied to the input and the scalar value
        /// </summary>
        IBinaryOp<T> Operation { get; }
        /// <summary>
        /// The value used in the operation
        /// </summary>
        T Value { get; }
    }

    /// <summary>
    /// A scalar operation, that is a single binary operation with a scalar value embedded
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    /// <typeparam name="C">The operation type</typeparam>
    public struct ScalarOp<T, C> : IUnaryOp<T>, ScalarAccess<T> where C : IBinaryOp<T>
    {
        /// <summary>
        /// The operation
        /// </summary>
        private C m_op;
        /// <summary>
        /// The scalar value
        /// </summary>
        private T m_value;

        /// <summary>
        /// Constructs a new scalar operation
        /// </summary>
        /// <param name="value">The scalar value</param>
        /// <param name="op">The binary operation</param>
        public ScalarOp(T value, C op)
        {
            m_value = value;
            m_op = op;
        }

        /// <summary>
        /// Executes the binary operation with the scalar value and the input
        /// </summary>
        /// <param name="value">The input value</param>
        /// <returns>The results of applying the operation to the scalar value and the input</returns>
        public T Op(T value) { return m_op.Op(value, m_value); }

        /// <summary>
        /// Hidden implementation of the ScalarAccess interface
        /// </summary>
        IBinaryOp<T> ScalarAccess<T>.Operation { get { return m_op; } }
        /// <summary>
        /// Hidden implementation of the ScalarAccess interface
        /// </summary>
        T ScalarAccess<T>.Value { get { return m_value; } }
    }

    public static partial class UFunc
    {
        /// <summary>
        /// The inner execution of a <see cref="T:NumCIL.IBinaryOp{0}"/>.
        /// This method will determine if the accessor is a <see cref="T:NumCIL.Generic.ILazyAccessor{0}"/>,
        /// and defer execution. Otherwise the binary flush operation is called
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary<T, C>(C op, NdArray<T> in1, NdArray<T> in2, ref NdArray<T> @out)
            where C : IBinaryOp<T>
        {
            if (@out.m_data is ILazyAccessor<T>)
                ((ILazyAccessor<T>)@out.m_data).AddOperation(op, @out, in1, in2);
            else
                UFunc_Op_Inner_Binary_Flush<T, C>(op, in1, in2, ref @out);
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
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_Flush<T, C>(C op, NdArray<T> in1, NdArray<T> in2, ref NdArray<T> @out)
            where C : IBinaryOp<T>
        {
            T[] d1 = in1.Data;
            T[] d2 = in2.Data;
            T[] d3 = @out.Data;

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
        /// The inner execution of a <see cref="T:NumCIL.IUnaryOp{0}"/>.
        /// This will just call the UnaryConv flush operation with Ta and Tb set to T
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush<T, C>(C op, NdArray<T> in1, ref NdArray<T> @out)
            where C : struct, IUnaryOp<T>
        {
            UFunc_Op_Inner_UnaryConv_Flush<T, T, C>(op, in1, ref @out);
        }

        /// <summary>
        /// The inner execution of a <see cref="T:NumCIL.IUnaryOp{0}"/>.
        /// This method will determine if the accessor is a <see cref="T:NumCIL.Generic.ILazyAccessor{0}"/>,
        /// and defer execution. Otherwise the unary flush operation is called
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        public static void UFunc_Op_Inner_Unary<T, C>(C op, NdArray<T> in1, ref NdArray<T> @out)
            where C : struct, IUnaryOp<T>
        {
            if (@out.m_data is ILazyAccessor<T>)
                ((ILazyAccessor<T>)@out.m_data).AddOperation(op, @out, in1);
            else
                UFunc_Op_Inner_Unary_Flush<T, C>(op, in1, ref @out);
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
        private static void UFunc_Op_Inner_UnaryConv<Ta, Tb, C>(NdArray<Ta> in1, ref NdArray<Tb> @out)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            UFunc_Op_Inner_UnaryConv_Flush<Ta, Tb, C>(new C(), in1, ref @out);
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
        private static void UFunc_Op_Inner_UnaryConv_Flush<Ta, Tb, C>(C op, NdArray<Ta> in1, ref NdArray<Tb> @out)
            where C : IUnaryConvOp<Ta, Tb>
        {
            Ta[] d1 = in1.Data;
            Tb[] d2 = @out.Data;

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
        /// The inner execution of a <see cref="T:NumCIL.INullaryOp{0}"/>.
        /// This method will determine if the accessor is a <see cref="T:NumCIL.Generic.ILazyAccessor{0}"/>,
        /// and defer execution. Otherwise the nullary flush operation is called
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Nullary<T, C>(C op, NdArray<T> @out)
            where C : INullaryOp<T>
        {
            if (@out.m_data is ILazyAccessor<T>)
                ((ILazyAccessor<T>)@out.m_data).AddOperation(op, @out);
            else
                UFunc_Op_Inner_Nullary_Flush<T, C>(op, @out);
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
            where C : INullaryOp<T>
        {
            T[] d = @out.Data;

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
