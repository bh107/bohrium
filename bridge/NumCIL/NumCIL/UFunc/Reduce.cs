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
    /// <summary>
    /// Universal function implementations (elementwise operations)
    /// </summary>
    public partial class UFunc
    {
        /// <summary>
        /// Wrapper class to represent a pending reduce operation in a list of pending operations
        /// </summary>
        /// <typeparam name="T">The type of data being processed</typeparam>
        public struct LazyReduceOperation<T> : IOp<T>
        {
            /// <summary>
            /// The axis to reduce
            /// </summary>
            public readonly long Axis;
            /// <summary>
            /// The operation to use for reduction
            /// </summary>
            public readonly IBinaryOp<T> Operation;

            /// <summary>
            /// Initializes a new instance of the <see cref="LazyReduceOperation&lt;T&gt;"/> struct.
            /// </summary>
            /// <param name="operation">The operation to reduce with</param>
            /// <param name="axis">The axis to reduce over</param>
            public LazyReduceOperation(IBinaryOp<T> operation, long axis) 
            {
                Operation = operation;
                Axis = axis; 
            }

            /// <summary>
            /// Required interface member that is not used
            /// </summary>
            /// <param name="a">Unused</param>
            /// <param name="b">Unused</param>
            /// <returns>Throws exception</returns>
            public T Op(T a, T b)
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Sets up the output array if it is null, or verifies it if it is supplied
        /// </summary>
        /// <typeparam name="T">The type of data to work with</typeparam>
        /// <param name="in1">The array to reduce</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output array</param>
        /// <returns>A correctly shaped output array or throws an exception</returns>
        private static NdArray<T> SetupReduceHelper<T>(NdArray<T> in1, long axis, NdArray<T> @out)
        {
            long j = 0;
            long[] dims = in1.Shape.Dimensions.Where(x => j++ != axis).Select(x => x.Length).ToArray();
            if (dims.LongLength == 0)
                dims = new long[] { 1 };

            if (@out == null)
            {
                //We allocate a new array with the appropriate dimensions
                @out = new NdArray<T>(dims);
            }
            else
            {
                if (@out.Shape.Dimensions.LongLength != dims.LongLength)
                    throw new Exception("Target array does not have the right number of dimensions");

                for (long i = 0; i < @out.Shape.Dimensions.LongLength; i++)
                    if (@out.Shape.Dimensions[i].Length != dims[i])
                        throw new Exception("Dimension size of target array is incorrect");
            }

            return @out;
        }

        /// <summary>
        /// Reduces the input argument on the specified axis
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        public static NdArray<T> Reduce<T, C>(NdArray<T> in1, long axis = 0, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            return Reduce_Entry<T, C>(new C(), in1, axis, @out);
        }

        /// <summary>
        /// The entry function for a reduction.
        /// This method will determine if the accessor is a <see cref="T:NumCIL.Generic.ILazyAccessor{0}"/>,
        /// and defer execution by wrapping it in a <see cref="T:NumCIL.UFunc.LazyReduceOperation{0}"/>. 
        /// Otherwise the reduce flush function is called
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<T> Reduce_Entry<T, C>(C op, NdArray<T> in1, long axis = 0, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            NdArray<T> v = SetupReduceHelper<T>(in1, axis, @out);

            if (v.DataAccessor is ILazyAccessor<T>)
                ((ILazyAccessor<T>)v.DataAccessor).AddOperation(new LazyReduceOperation<T>(new C(), axis), v, in1);
            else
                ApplyManager.ApplyReduce<T, C>(op, axis, in1, v);

            return v;
        }

        /// <summary>
        /// Reduces the input argument on the specified axis
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="op">The operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        public static NdArray<T> Reduce<T>(IBinaryOp<T> op, NdArray<T> in1, long axis = 0, NdArray<T> @out = null)
        {
            var method = typeof(UFunc).GetMethod("Reduce_Entry", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(T), op.GetType());
            return (NdArray<T>)gm.Invoke(null, new object[] { op, in1, axis, @out });
        }

		/// <summary>
		/// Attempts to use a typed version of the Reduce call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="T">The type of data to operate on</typeparam>
		/// <typeparam name="C">The type of operation to reduce with</typeparam>
		/// <param name="op">The instance of the operation to reduce with</param>
		/// <param name="in1">The input argument</param>
		/// <param name="axis">The axis to reduce</param>
		/// <param name="out">The output target</param>
		/// <returns>The output target</returns>		
		private static bool UFunc_Reduce_Inner_Flush_Typed<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(T).FullName + "#" + op.GetType().FullName + "#RED";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(UFunc).GetMethod("UFunc_Reduce_Inner_Flush", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					axis.GetType(),
					in1.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { typeof(T), op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, axis, in1, @out });
				return true;
			}

			return false;
		}

    }
}
