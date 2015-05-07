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
		/// Lookup table for generic methods that have been created and are ready for use
		/// </summary>
		private static Dictionary<string, System.Reflection.MethodInfo> _resolvedMethods = new Dictionary<string, System.Reflection.MethodInfo>();

		/// <summary>
		/// Attempts to use a typed version of the BinaryConv call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="Ta">The type of input data to operate on</typeparam>
		/// <typeparam name="Tb">The type of output data to operate on</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="in2">The right-hand-side input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_BinaryConv_Flush_Typed<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#BIN";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(UFunc).GetMethod("UFunc_Op_Inner_BinaryConv_Flush", 
				System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
				null,
				new Type[] {
					op.GetType(),
					in1.GetType(),
					in2.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { typeof(Ta), typeof(Tb), op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, in1, in2, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the BinaryConv call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="Ta">The type of input data to operate on</typeparam>
		/// <typeparam name="Tb">The type of output data to operate on</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="in2">The right-hand-side input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_BinaryConv_LhsScalar_Flush_Typed<Ta, Tb, C>(C op, Ta scalar, NdArray<Ta> in2, NdArray<Tb> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#LHS";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(UFunc).GetMethod("UFunc_Op_Inner_BinaryConv_LhsScalar_Flush", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					scalar.GetType(),
					in2.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { typeof(Ta), typeof(Tb), op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, scalar, in2, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the BinaryConv call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="Ta">The type of input data to operate on</typeparam>
		/// <typeparam name="Tb">The type of output data to operate on</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The left-hand-side input argument</param>
		/// <param name="in2">The right-hand-side input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_BinaryConv_RhsScalar_Flush_Typed<Ta, Tb, C>(C op, NdArray<Ta> in1, Ta scalar, NdArray<Tb> @out)
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#RHS";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(UFunc).GetMethod("UFunc_Op_Inner_BinaryConv_RhsScalar_Flush", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					in1.GetType(),
					scalar.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { typeof(Ta), typeof(Tb), op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, in1, scalar, @out });
				return true;
			}

			return false;
		}

		/// <summary>
		/// Attempts to use a typed version of the UnaryConv call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="Ta">The type of input data to operate on</typeparam>
		/// <typeparam name="Tb">The type of output data to generate</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_UnaryConv_Flush_Typed<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out)
			where C : IUnaryConvOp<Ta, Tb>
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "##UN";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(UFunc).GetMethod("UFunc_Op_Inner_UnaryConv_Flush", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					in1.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { typeof(Ta), typeof(Tb), op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, in1, @out });
				return true;
			}

			return false;			
		}

		/// <summary>
		/// Attempts to use a typed version of the UnaryConv call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="Ta">The type of input data to operate on</typeparam>
		/// <typeparam name="Tb">The type of output data to generate</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="in1">The input argument</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_UnaryConv_Flush_Scalar_Typed<Ta, Tb, C>(C op, Ta scalar, NdArray<Tb> @out)
			where C : IUnaryConvOp<Ta, Tb>
		{
			System.Reflection.MethodInfo f;
			var key = typeof(Ta).FullName + "#" + typeof(Tb).FullName + "#" + op.GetType().FullName + "#SCL";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(UFunc).GetMethod("UFunc_Op_Inner_UnaryConv_Flush_Scalar", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					scalar.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { typeof(Ta), typeof(Tb), op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, scalar, @out });
				return true;
			}

			return false;			
		}

		/// <summary>
		/// Attempts to use a typed version of the NullaryImpl call, 
		/// to avoid dependency on the JIT being able to inline struct methods
		/// </summary>
		/// <typeparam name="T">The type of data to generat</typeparam>
		/// <typeparam name="C">The type of operation to perform</typeparam>
		/// <param name="op">The operation instance</param>
		/// <param name="out">The output target</param>
		private static bool UFunc_Op_Inner_Nullary_Impl_Flush_Typed<T, C>(C op, NdArray<T> @out)
			where C : struct, INullaryOp<T>
		{
			System.Reflection.MethodInfo f;
			var key = typeof(T).FullName + "#" + op.GetType().FullName + "#NULL";
			if (!_resolvedMethods.TryGetValue(key, out f))
			{
				var n = typeof(UFunc).GetMethod("UFunc_Op_Inner_Nullary_Impl_Flush", 
					System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static,
					null,
					new Type[] {
					op.GetType(),
					@out.GetType()
				}, null);

				_resolvedMethods[key] = f = n == null ? null : n.MakeGenericMethod(new Type[] { typeof(T), op.GetType() });
			}

			if (f != null)
			{
				f.Invoke(null, new object[] { op, @out });
				return true;
			}			

			return false;
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
        private static void UFunc_Op_Inner_Binary_Flush<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (in1.DataAccessor.Length == 1 && in1.DataAccessor.GetType() == typeof(DefaultAccessor<T>))
                UFunc_Op_Inner_Binary_LhsScalar_Flush(op, in1.DataAccessor[0], in2, @out);
            else if (in2.DataAccessor.Length == 1 && in2.DataAccessor.GetType() == typeof(DefaultAccessor<T>))
                UFunc_Op_Inner_Binary_RhsScalar_Flush(op, in1, in2.DataAccessor[0], @out);
            else
            {
                if (UnsafeAPI.UFunc_Op_Inner_Binary_Flush_Unsafe(op, in1, in2, ref @out))
                    return;

                UFunc_Op_Inner_BinaryConv_Flush<T, T, C>(op, in1, in2, @out);
            }
        }


        /// <summary>
        /// Actually executes a binary operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IBinaryOp{0}"/> on each element.
        /// This implementation is optimized for use with up to 4 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="scalar">The left-hand-side scalar argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_LhsScalar_Flush<T, C>(C op, T scalar, NdArray<T> in2, NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (UnsafeAPI.UFunc_Op_Inner_Binary_LhsScalar_Flush_Unsafe(op, scalar, in2, ref @out))
                return;

            UFunc_Op_Inner_BinaryConv_LhsScalar_Flush<T, T, C>(op, scalar, in2, @out);
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
        /// <param name="scalar">The right-hand-side scalar argument</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Binary_RhsScalar_Flush<T, C>(C op, NdArray<T> in1, T scalar, NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (UnsafeAPI.UFunc_Op_Inner_Binary_RhsScalar_Flush_Unsafe(op, in1, scalar, ref @out))
                return;

            UFunc_Op_Inner_BinaryConv_RhsScalar_Flush<T, T, C>(op, in1, scalar, @out);
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
        private static void UFunc_Op_Inner_Unary_Flush<T, C>(C op, NdArray<T> in1, NdArray<T> @out)
            where C : struct, IUnaryOp<T>
        {
            if (in1.DataAccessor.Length == 1 && in1.DataAccessor.GetType() == typeof(DefaultAccessor<T>))
                UFunc_Op_Inner_Unary_Flush_Scalar<T, C>(op, in1.DataAccessor[0], @out);
            else
            {
                if (!UnsafeAPI.UFunc_Op_Inner_Unary_Flush_Unsafe<T, C>(op, in1, ref @out))
                    UFunc_Op_Inner_UnaryConv_Flush<T, T, C>(op, in1, @out);
            }
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
        private static void UFunc_Op_Inner_UnaryConv<Ta, Tb, C>(NdArray<Ta> in1, NdArray<Tb> @out)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            UFunc_Op_Inner_UnaryConv_Flush<Ta, Tb, C>(new C(), in1, @out);
        }


        /// <summary>
        /// The inner execution of a <see cref="T:NumCIL.IUnaryConvOp{0}"/>.
        /// This method will always call the unary conv flush method, because the lazy evaluation system does not implement support for handling conversion operations yet.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="scalar">The input scalar</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_UnaryConv_Scalar<Ta, Tb, C>(Ta scalar, NdArray<Tb> @out)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            UFunc_Op_Inner_UnaryConv_Flush_Scalar<Ta, Tb, C>(new C(), scalar, @out);
        }

		/// <summary>
        /// The inner execution of a <see cref="T:NumCIL.IUnaryOp{0}"/>.
        /// This will just call the UnaryConv flush operation with Ta and Tb set to T
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="scalar">The input scalar</param>
        /// <param name="out">The output target</param>
        private static void UFunc_Op_Inner_Unary_Flush_Scalar<T, C>(C op, T scalar, NdArray<T> @out)
            where C : struct, IUnaryOp<T>
        {
            if (!UnsafeAPI.UFunc_Op_Inner_Unary_Scalar_Flush_Unsafe<T, C>(op, scalar, ref @out))
                UFunc_Op_Inner_UnaryConv_Flush_Scalar<T, T, C>(op, scalar, @out);
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
			where C : struct, INullaryOp<T>
		{
			if (!UnsafeAPI.UFunc_Op_Inner_Nullary_Flush_Unsafe<T, C>(op, @out))
				UFunc_Op_Inner_Nullary_Impl_Flush<T, C>(op, @out);


		}
    }
}
