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
