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
    public partial class UFunc
    {
        /// <summary>
        /// Setup function for shaping the input and output arrays to broadcast compatible shapes.
        /// If no output array is given, a compatible output array is created
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to operate on</typeparam>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>A tupple with broadcast compatible shapes for the inputs, and an output array</returns>
        private static Tuple<NdArray<Ta>, NdArray<Ta>, NdArray<Tb>> SetupApplyHelper<Ta, Tb>(NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out)
        {
            Tuple<Shape, Shape> broadcastshapes = Shape.ToBroadcastShapes(in1.Shape, in2.Shape);
            if (@out == null)
            {
                //We allocate a new array
                @out = new NdArray<Tb>(broadcastshapes.Item1.Plain);
            }
            else
            {
                if (@out.Shape.Dimensions.Length != broadcastshapes.Item1.Dimensions.Length)
                    throw new Exception("Target array does not have the right number of dimensions");

                for (long i = 0; i < @out.Shape.Dimensions.Length; i++)
                    if (@out.Shape.Dimensions[i].Length != broadcastshapes.Item1.Dimensions[i].Length)
                        throw new Exception("Dimension size of target array is incorrect");
            }

            var op1 = in1.Reshape(broadcastshapes.Item1);
            var op2 = in2.Reshape(broadcastshapes.Item2);

            return new Tuple<NdArray<Ta>, NdArray<Ta>, NdArray<Tb>>(op1, op2, @out);
        }

        /// <summary>
        /// Setup function for shaping the input and output arrays to broadcast compatible shapes.
        /// If no output array is given, a compatible output array is created
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <param name="in1">The input data</param>
        /// <param name="out">The output target</param>
        /// <returns>A compatible output array or throws an exception</returns>
        private static NdArray<Tb> SetupApplyHelper<Ta, Tb>(NdArray<Ta> in1, NdArray<Tb> @out)
        {
            if (@out == null)
            {
                //We allocate a new array
                @out = new NdArray<Tb>(in1.Shape.Plain);
            }
            else
            {
                if (@out.Shape.Dimensions.Length != in1.Shape.Dimensions.Length)
                    throw new Exception("Target array does not have the right number of dimensions");

                for (long i = 0; i < @out.Shape.Dimensions.Length; i++)
                    if (@out.Shape.Dimensions[i].Length != in1.Shape.Dimensions[i].Length)
                        throw new Exception("Dimension size of target array is incorrect");
            }

            return @out;
        }

        /// <summary>
        /// Function that is used as the entry point for applying a binary operator.
        /// It will setup the output array and then call the evaluation method
        /// It has a unique name because it is faster to look up the method through reflection,
        /// if there is only one version of it.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The operation type to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        private static NdArray<T> Apply_Entry_Binary<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            Tuple<NdArray<T>, NdArray<T>, NdArray<T>> v = SetupApplyHelper(in1, in2, @out);
            @out = v.Item3;

            if (@out.DataAccessor is ILazyAccessor<T>)
                ((ILazyAccessor<T>)@out.DataAccessor).AddOperation(op, @out, v.Item1, v.Item2);
            else
                ApplyManager.ApplyBinaryOp<T, C>(op, v.Item1, v.Item2, @out);

            return @out;
        }

        /// <summary>
        /// Function that is used as the entry point for applying a binary conversion operator.
        /// It will setup the output array and then call the evaluation method
        /// It has a unique name because it is faster to look up the method through reflection,
        /// if there is only one version of it.
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to operate on</typeparam>
        /// <typeparam name="C">The operation type to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        private static NdArray<Tb> Apply_Entry_BinaryConv<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out = null)
            where C : struct, IBinaryConvOp<Ta, Tb>
        {
            Tuple<NdArray<Ta>, NdArray<Ta>, NdArray<Tb>> v = SetupApplyHelper(in1, in2, @out);
            @out = v.Item3;

            if (@out.DataAccessor is ILazyAccessor<Tb>)
                ((ILazyAccessor<Tb>)@out.DataAccessor).AddConversionOperation(op, @out, v.Item1, v.Item2);
            else
                ApplyManager.ApplyBinaryConvOp<Ta, Tb, C>(op, v.Item1, v.Item2, @out);

            return @out;
        }
        /// <summary>
        /// Applies the operation to the input arrays
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        public static NdArray<T> Apply<T, C>(NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            return Apply_Entry_Binary<T, C>(new C(), in1, in2, @out);
        }

        /// <summary>
        /// Applies the operation to the input arrays
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        public static NdArray<Tb> Apply<Ta, Tb, C>(NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out = null)
            where C : struct, IBinaryConvOp<Ta, Tb>
        {
            return Apply_Entry_BinaryConv<Ta, Tb, C>(new C(), in1, in2, @out);
        }

        /// <summary>
        /// Applies the operation to the input arrays
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="op">The operation instance to use</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        public static NdArray<T> Apply<T>(IBinaryOp<T> op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
        {
            var method = typeof(UFunc).GetMethod("Apply_Entry_Binary", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(T), op.GetType());
            return (NdArray<T>)gm.Invoke(null, new object[] { op, in1, in2, @out });
        }

        /// <summary>
        /// Applies the operation to the input arrays
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to operate on</typeparam>
        /// <param name="op">The operation instance to use</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        public static NdArray<Tb> Apply<Ta, Tb>(IBinaryConvOp<Ta, Tb> op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out = null)
        {
            var method = typeof(UFunc).GetMethod("Apply_Entry_BinaryConv", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(Ta), typeof(Tb), op.GetType());
            return (NdArray<Tb>)gm.Invoke(null, new object[] { op, in1, in2, @out });
        }

        /// <summary>
        /// Applies the operation to the input array
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        public static NdArray<T> Apply<T, C>(NdArray<T> in1, NdArray<T> @out = null)
            where C : struct, IUnaryOp<T>
        {
            return Apply_Entry_Unary<T, C>(new C(), in1, @out);
        }

        /// <summary>
        /// Function that is used as the entry point for applying a unary operator.
        /// It will setup the output array and then call the evaluation method
        /// It has a unique name because it is faster to look up the method through reflection,
        /// if there is only one version of it.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The operation type to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        private static NdArray<T> Apply_Entry_Unary<T, C>(C op, NdArray<T> in1, NdArray<T> @out = null)
            where C : struct, IUnaryOp<T>
        {
            NdArray<T> v = SetupApplyHelper(in1, @out);

            if (v.DataAccessor is ILazyAccessor<T>)
                ((ILazyAccessor<T>)v.DataAccessor).AddOperation(op, v, in1);
            else
                ApplyManager.ApplyUnaryOp<T, C>(op, in1, v);

            return v;
        }

        /// <summary>
        /// Applies the operation to the input array
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="op">The operation instance to use</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        public static NdArray<T> Apply<T>(IUnaryOp<T> op, NdArray<T> in1, NdArray<T> @out = null)
        {
            var method = typeof(UFunc).GetMethod("Apply_Entry_Unary", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(T), op.GetType());
            return (NdArray<T>)gm.Invoke(null, new object[] { op, in1, @out });
        }

        /// <summary>
        /// Applies the operation to the input array
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <typeparam name="C">The operation type to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        public static NdArray<Tb> Apply<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out = null)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            return Apply_Entry_Unary_Conv<Ta, Tb, C>(op, in1, @out);
        }

        /// <summary>
        /// Applies the operation to the input array
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <typeparam name="C">The operation type to perform</typeparam>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        public static NdArray<Tb> Apply<Ta, Tb, C>(NdArray<Ta> in1, NdArray<Tb> @out = null)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            return Apply_Entry_Unary_Conv<Ta, Tb, C>(new C(), in1, @out);
        }

        /// <summary>
        /// Applies the operation to the input array
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        public static NdArray<Tb> Apply<Ta, Tb>(IUnaryConvOp<Ta, Tb> op, NdArray<Ta> in1, NdArray<Tb> @out = null)
        {
            var method = typeof(UFunc).GetMethod("Apply_Entry_UnaryConv", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(Ta), typeof(Tb), op.GetType());
            return (NdArray<Tb>)gm.Invoke(null, new object[] { op, in1, @out });
        }

        /// <summary>
        /// Applies the operation to the input array
        /// </summary>
        /// <typeparam name="Ta">The type of input data to operate on</typeparam>
        /// <typeparam name="Tb">The type of output data to generate</typeparam>
        /// <typeparam name="C">The operation type to perform</typeparam>
        /// <param name="op">The operation to perform</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        private static NdArray<Tb> Apply_Entry_Unary_Conv<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out = null)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            NdArray<Tb> v = SetupApplyHelper(in1, @out);

            if (v.DataAccessor is ILazyAccessor<Tb>)
                ((ILazyAccessor<Tb>)v.DataAccessor).AddConversionOperation<Ta>(op, v, in1);
            else
                ApplyManager.ApplyUnaryConvOp<Ta, Tb, C>(op, in1, v);

            return v;
        }

        /// <summary>
        /// Applies the lambda operation to the input arrays
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="op">The lambda function to apply</param>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        public static NdArray<T> Apply<T>(Func<T, T, T> op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
        {
            return Apply_Entry_Binary<T, BinaryLambdaOp<T>>(op, in1, in2, @out);
        }

        /// <summary>
        /// Applies the lambda operation to the input array
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="op">The lambda function to apply</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        public static NdArray<T> Apply<T>(Func<T, T> op, NdArray<T> in1, NdArray<T> @out = null)
        {
            //TODO: Should attempt to compile a new struct with the lambda function embedded to avoid the virtual function call overhead
            return Apply_Entry_Unary<T, UnaryLambdaOp<T>>(op, in1, @out);
        }

        /// <summary>
        /// Applies the operation to the output array
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The function to apply</param>
        /// <param name="out">The output target</param>
        private static void Apply_Entry_Nullary<T, C>(C op, NdArray<T> @out)
            where C : struct, INullaryOp<T>
        {
            if (@out.DataAccessor is ILazyAccessor<T>)
                ((ILazyAccessor<T>)@out.DataAccessor).AddOperation(op, @out);
            else
                ApplyManager.ApplyNullaryOp<T, C>(op, @out);
        }

        /// <summary>
        /// Applies the operation to the output array
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The function to apply</param>
        /// <param name="out">The output target</param>
        public static void Apply<T, C>(C op, NdArray<T> @out)
            where C : struct, INullaryOp<T>
        {
            Apply_Entry_Nullary<T, C>(op, @out);
        }

        /// <summary>
        /// Applies the operation to the output array
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="op">The function to apply</param>
        /// <param name="out">The output target</param>
        public static void Apply<T>(INullaryOp<T> op, NdArray<T> @out)
        {
            var method = typeof(UFunc).GetMethod("Apply_Entry_Nullary", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(T), op.GetType());
            gm.Invoke(null, new object[] { op, @out });
        }
    }
}