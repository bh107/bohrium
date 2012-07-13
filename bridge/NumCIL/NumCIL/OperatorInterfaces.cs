using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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
    public interface IUnaryConvOp<Ta, Tb> : IOp<Tb>
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
    public interface INullaryOp<T> : IOp<T>
    {
        /// <summary>
        /// Performs an operation
        /// </summary>
        /// <returns>The result of the operation</returns>
        T Op();
    }

    /// <summary>
    /// Interface to allow reading the scalar value from a ScalarOp.
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface IScalarAccess<T>
    {
        /// <summary>
        /// The operation applied to the input and the scalar value
        /// </summary>
        IOp<T> Operation { get; }
        /// <summary>
        /// The value used in the operation
        /// </summary>
        T Value { get; }
    }

    /// <summary>
    /// Interface that enables the implementation to differentiate between
    /// left-hand-side and right-hand-side scalar operands in binary operations
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface IScalarAccessBinary<T> : IScalarAccess<T>
    {
        /// <summary>
        /// Gets a value indicating if the scalar is a left hand side operand
        /// </summary>
        bool IsLhsOperand { get; }
    }
}
