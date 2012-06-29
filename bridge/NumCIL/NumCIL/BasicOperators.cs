using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL
{
    /// <summary>
    /// An operation that outputs the same value for each input
    /// </summary>
    /// <typeparam name="T">The type of data to produce</typeparam>
    public struct GenerateOp<T> : INullaryOp<T>, IScalarAccess<T>
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

        /// <summary>
        /// Returns the operation performed
        /// </summary>
        IOp<T> IScalarAccess<T>.Operation
        {
            get { return new CopyOp<T>(); }
        }

        /// <summary>
        /// Returns the value to set
        /// </summary>
        T IScalarAccess<T>.Value
        {
            get { return Value; }
        }
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
    /// A scalar operation, that is a single binary operation with a scalar value embedded
    /// The operation is performed with the scalar value as the right-hand-side argument
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    /// <typeparam name="C">The operation type</typeparam>
    public struct RhsScalarOp<T, C> : IUnaryOp<T>, IScalarAccessBinary<T> where C : struct, IBinaryOp<T>
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
        public RhsScalarOp(T value, C op)
        {
            m_value = value;
            m_op = op;
        }

        /// <summary>
        /// Executes the binary operation with the scalar value and the input.
        /// </summary>
        /// <param name="value">The input value</param>
        /// <returns>The results of applying the operation to the scalar value and the input</returns>
        public T Op(T value) { return m_op.Op(value, m_value); }

        /// <summary>
        /// Hidden implementation of the ScalarAccess interface
        /// </summary>
        IOp<T> IScalarAccess<T>.Operation { get { return m_op; } }
        /// <summary>
        /// Hidden implementation of the ScalarAccess interface
        /// </summary>
        T IScalarAccess<T>.Value { get { return m_value; } }
        /// <summary>
        /// Hidden implemenation of the IScalarAccessBinary interface
        /// </summary>
        bool IScalarAccessBinary<T>.IsLhsOperand { get { return false; } }
    }

    /// <summary>
    /// A scalar operation, that is a single binary operation with a scalar value embedded.
    /// The operation is performed with the scalar value as the left-hand-side argument.
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    /// <typeparam name="C">The operation type</typeparam>
    public struct LhsScalarOp<T, C> : IUnaryOp<T>, IScalarAccessBinary<T> where C : struct, IBinaryOp<T>
    {
        /// <summary>
        /// The operation
        /// </summary>
        private readonly C m_op;
        /// <summary>
        /// The scalar value
        /// </summary>
        private readonly T m_value;

        /// <summary>
        /// Constructs a new scalar operation
        /// </summary>
        /// <param name="value">The scalar value</param>
        /// <param name="op">The binary operation</param>
        public LhsScalarOp(T value, C op)
        {
            m_value = value;
            m_op = op;
        }

        /// <summary>
        /// Executes the binary operation with the scalar value and the input
        /// </summary>
        /// <param name="value">The input value</param>
        /// <returns>The results of applying the operation to the scalar value and the input</returns>
        public T Op(T value) { return m_op.Op(m_value, value); }

        /// <summary>
        /// Hidden implementation of the IScalarAccess interface
        /// </summary>
        IOp<T> IScalarAccess<T>.Operation { get { return m_op; } }
        /// <summary>
        /// Hidden implementation of the IScalarAccess interface
        /// </summary>
        T IScalarAccess<T>.Value { get { return m_value; } }
        /// <summary>
        /// Hidden implemenation of the IScalarAccessBinary interface
        /// </summary>
        bool IScalarAccessBinary<T>.IsLhsOperand { get { return true; } }
    }

    /// <summary>
    /// A scalar operation, that is a single binary operation with a scalar value embedded
    /// The operation is performed with the scalar value as the right-hand-side argument
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    /// <typeparam name="C">The operation type</typeparam>
    public struct ScalarValue<T, C> : INullaryOp<T>, IScalarAccess<T> where C : IUnaryOp<T>
    {
        /// <summary>
        /// The operation
        /// </summary>
        private readonly C m_op;
        /// <summary>
        /// The scalar value
        /// </summary>
        private readonly T m_value;

        /// <summary>
        /// Constructs a new scalar operation
        /// </summary>
        /// <param name="value">The scalar value</param>
        /// <param name="op">The binary operation</param>
        public ScalarValue(T value, C op)
        {
            m_value = value;
            m_op = op;
        }

        /// <summary>
        /// Executes the binary operation with the scalar value and the input.
        /// </summary>
        /// <returns>The results of applying the operation to the scalar value and the input</returns>
        public T Op() { return m_value; }

        /// <summary>
        /// Hidden implementation of the ScalarAccess interface
        /// </summary>
        IOp<T> IScalarAccess<T>.Operation { get { return m_op; } }
        /// <summary>
        /// Hidden implementation of the ScalarAccess interface
        /// </summary>
        T IScalarAccess<T>.Value { get { return m_value; } }
    }
}
