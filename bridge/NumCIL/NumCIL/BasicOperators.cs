#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium:
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

namespace NumCIL
{
    /// <summary>
    /// An operation that outputs the same value for each input
    /// </summary>
    /// <typeparam name="T">The type of data to produce</typeparam>
    public struct GenerateOp<T> : INullaryOp<T>, NumCIL.Generic.Operators.ICopyOperation
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
    public struct CopyOp<T> : IUnaryOp<T>, NumCIL.Generic.Operators.ICopyOperation
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
}
