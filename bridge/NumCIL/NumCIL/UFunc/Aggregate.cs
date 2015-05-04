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
    public static partial class UFunc
    {
        /// <summary>
        /// Wrapper class to represent a pending reduce operation in a list of pending operations
        /// </summary>
        /// <typeparam name="T">The type of data being processed</typeparam>
        public struct LazyAggregateOperation<T> : IOp<T>
        {
            /// <summary>
            /// The operation to use for reduction
            /// </summary>
            public readonly IBinaryOp<T> Operation;

            /// <summary>
            /// Initializes a new instance of the <see cref="LazyReduceOperation&lt;T&gt;"/> struct.
            /// </summary>
            /// <param name="operation">The operation to reduce with</param>
            public LazyAggregateOperation(IBinaryOp<T> operation)
            {
                Operation = operation;
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
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="T">The value to operate on</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        public static T Aggregate<T>(IBinaryOp<T> op, NdArray<T> in1)
        {
            var method = typeof(UFunc).GetMethod("Aggregate_Entry", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(T), op.GetType());
            return (T)gm.Invoke(null, new object[] { op, in1 });
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="T">The value to operate on</typeparam>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        public static T Aggregate<T, C>(NdArray<T> in1)
            where C : struct, IBinaryOp<T>
        {
            return Aggregate_Entry<T, C>(new C(), in1);
        }

        /// <summary>
        /// Calculates the scalar result of applying the binary operation to all elements
        /// </summary>
        /// <typeparam name="T">The value to operate on</typeparam>
        /// <typeparam name="C">The operation to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The array to aggregate</param>
        /// <returns>A scalar value that is the result of aggregating all elements</returns>
        private static T Aggregate_Entry<T, C>(C op, NdArray<T> in1)
            where C : struct, IBinaryOp<T>
        {
            if (in1.DataAccessor is ILazyAccessor<T>)
            {
                var v = new NumCIL.Generic.NdArray<T>((IDataAccessor<T>)Activator.CreateInstance(in1.DataAccessor.GetType(), 1L));
                ((ILazyAccessor<T>)v.DataAccessor).AddOperation(new LazyAggregateOperation<T>(new C()), v, in1);
                return v.Value[0];
            }
            else
            {
                T result;
                ApplyManager.ApplyAggregate<T, C>(op, in1, out result);
                return result;
            }
        }

		/// <summary>
		/// Attempts to use a typed version of the Aggregate call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="T">The type of data to operate on</typeparam>
		/// <typeparam name="C">The type of operation to reduce with</typeparam>
		/// <param name="op">The instance of the operation to reduce with</param>
		/// <param name="in1">The input argument</param>
		/// <param name="axis">The axis to reduce</param>
		/// <param name="out">The output target</param>
		/// <returns>The output target</returns>		
		private static bool UFunc_Aggregate_Inner_Flush_Typed<T, C>(C op, NdArray<T> in1, out T @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(T).FullName + "#" + op.GetType().FullName + "#AGR";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(UFunc).GetMethod("UFunc_Aggregate_Inner_Flush", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					in1.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { typeof(T), op.GetType() });
			}

			if (f != null)
			{
				@out = (T)f.Invoke(null, new object[] { op, in1 });
				return true;
			}
			else
				@out = default(T);

			return false;
		}
    }
}
