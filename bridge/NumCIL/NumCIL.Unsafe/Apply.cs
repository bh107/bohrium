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
using System.Runtime.InteropServices;

namespace NumCIL.Unsafe
{
    internal static partial class Apply
    {
		/// <summary>
		/// Lookup table for generic methods that have been created and are ready for use
		/// </summary>
		private static Dictionary<string, System.Reflection.MethodInfo> _resolvedMethods = new Dictionary<string, System.Reflection.MethodInfo>();

		/// <summary>
		/// Attempts to use a typed version of the Unary call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="Ta">The type of input data to operate on</typeparam>
		/// <typeparam name="Tb">The type of output data to operate on</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_Unary_Flush_Typed<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#UN";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(Apply).GetMethod("UFunc_Op_Inner_Unary_Flush_" + typeof(Ta).Name + "_TypedImpl",
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					in1.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, in1, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the Unary call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="Ta">The type of input data to operate on</typeparam>
		/// <typeparam name="Tb">The type of output data to operate on</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="scalar">The left-hand-side input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_Unary_Scalar_Flush_Typed<Ta, Tb, C>(C op, Ta scalar, NdArray<Tb> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#SCL";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(Apply).GetMethod("UFunc_Op_Inner_Unary_Scalar_Flush_" + typeof(Ta).Name + "_TypedImpl", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					scalar.GetType(),
					@out.GetType()
				}, null );

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, scalar, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the Nullary call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="T">The type of output data to operate on</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_Nullary_Flush_Typed<T, C>(C op, NdArray<T> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(T).FullName + "#" + op.GetType().FullName + "#NULL";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(Apply).GetMethod("UFunc_Op_Inner_Nullary_Flush_" + typeof(T).Name + "_TypedImpl",
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the Binary call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="in2">The right-hand-side input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_Binary_Flush_Typed<Ta, Tb, C>(C op, NdArray<Ta> in1,  NdArray<Ta> in2, NdArray<Tb> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#BIN";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(Apply).GetMethod("UFunc_Op_Inner_Binary_Flush_" + typeof(Ta).Name + "_TypedImpl", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					in1.GetType(),
					in2.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, in1, in2, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the Binary call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="in2">The right-hand-side input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_Binary_LhsScalar_Flush_Typed<Ta, Tb, C>(C op, Ta scalar,  NdArray<Ta> in2, NdArray<Tb> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#LHS";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(Apply).GetMethod("UFunc_Op_Inner_Binary_LhsScalar_Flush_" + typeof(Ta).Name + "_TypedImpl", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					scalar.GetType(),
					in2.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, scalar, in2, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the Binary call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="in2">The right-hand-side input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_Binary_RhsScalar_Flush_Typed<Ta, Tb, C>(C op, NdArray<Ta> in1, Ta scalar, NdArray<Tb> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#RHS";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(Apply).GetMethod("UFunc_Op_Inner_Binary_RhsScalar_Flush_" + typeof(Ta).Name + "_TypedImpl", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					in1.GetType(),
					scalar.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, in1, scalar, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the Binary call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="in2">The right-hand-side input argument</param>
		/// <param name="out">The output target</param>
		internal static bool UFunc_Reduce_Inner_Flush_Typed<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(T).FullName + "#" + op.GetType().FullName + "#RED";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(Reduce).GetMethod("UFunc_Reduce_Inner_Flush_" + typeof(T).Name + "_TypedImpl", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					axis.GetType(),
					in1.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, axis, in1, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the Binary call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="in2">The right-hand-side input argument</param>
		/// <param name="out">The output target</param>
		internal static bool UFunc_Aggregate_Entry_Typed<T, C>(C op, NdArray<T> in1, out T @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(T).FullName + "#" + op.GetType().FullName + "#AGR";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(Aggregate).GetMethod("Aggregate_Entry_" + typeof(T).Name + "_TypedImpl", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					in1.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { op.GetType() });
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