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
    public interface IBinaryOp<T> : IOp<T>, IBinaryConvOp<T, T> { }

    /// <summary>
    /// Describes an operation that takes two arguments and produce a boolean output
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface IBinaryCompareOp<T> : IBinaryConvOp<T, bool> { }

    /// <summary>
    /// Describes an operation that takes an input argument and produce an ouput
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface IUnaryOp<T> : IUnaryConvOp<T, T> { }

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
    /// Describes an operation that takes an input argument and produce an ouput
    /// </summary>
    /// <typeparam name="Ta">The input data type</typeparam>
    /// <typeparam name="Tb">The output data type</typeparam>
    public interface IBinaryConvOp<Ta, Tb> : IOp<Tb>
    {
        /// <summary>
        /// Performs the operation
        /// </summary>
        /// <param name="a">An input argument</param>
        /// <param name="b">An input argument</param>
        /// <returns>The converted value</returns>
        Tb Op(Ta a, Ta b);
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
}
