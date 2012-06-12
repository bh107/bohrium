using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL
{
    //Support for .Net 3.5
    /// <summary>
    /// Tupple data type for containing a single typesafe element
    /// </summary>
    /// <typeparam name="T1">The type of element 1</typeparam>
    public class Tuple<T1>
    {
        /// <summary>
        /// The first element in the tupple
        /// </summary>
        public T1 Item1;

        /// <summary>
        /// Initializes a new instance of the <see cref="Tuple&lt;T1&gt;"/> class.
        /// </summary>
        /// <param name="t1">The value of the first element</param>
        public Tuple(T1 t1) { Item1 = t1; }
    }

    /// <summary>
    /// Tupple data type for containing a two typesafe elements
    /// </summary>
    /// <typeparam name="T1">The type of element 1</typeparam>
    /// <typeparam name="T2">The type of element 2</typeparam>
    public class Tuple<T1, T2> : Tuple<T1> 
    { 
        /// <summary>
        /// The second element in the tupple
        /// </summary>
        public T2 Item2;

        /// <summary>
        /// Initializes a new instance of the <see cref="Tuple&lt;T1, T2&gt;"/> class.
        /// </summary>
        /// <param name="t1">The value of the first tupple element</param>
        /// <param name="t2">The value of the second tupple element</param>
        public Tuple(T1 t1, T2 t2)
            : base(t1)
        { Item2 = t2; }
    }

    /// <summary>
    /// Tupple data type for containing a three typesafe elements
    /// </summary>
    /// <typeparam name="T1">The type of element 1</typeparam>
    /// <typeparam name="T2">The type of element 2</typeparam>
    /// <typeparam name="T3">The type of element 3</typeparam>
    public class Tuple<T1, T2, T3> : Tuple<T1, T2>
    {
        /// <summary>
        /// The third element in the tupple
        /// </summary>
        public T3 Item3;

        /// <summary>
        /// Initializes a new instance of the <see cref="Tuple&lt;T1, T2, T3&gt;"/> class.
        /// </summary>
        /// <param name="t1">The value of the first tupple element</param>
        /// <param name="t2">The value of the second tupple element</param>
        /// <param name="t3">The value of the third tupple element</param>
        public Tuple(T1 t1, T2 t2, T3 t3)
            : base(t1, t2)
        { Item3 = t3; }
    }

    public partial class UFunc
    {
        /// <summary>
        /// Setup function for shaping the input and output arrays to broadcast compatible shapes.
        /// If no output array is given, a compatible output array is created
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="in2">The right-hand-side input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>A tupple with broadcast compatible shapes for the inputs, and an output array</returns>
        private static Tuple<Shape, Shape, NdArray<T>> SetupApplyHelper<T>(NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
        {
            Tuple<Shape, Shape> broadcastshapes = Shape.ToBroadcastShapes(in1.Shape, in2.Shape);
            if (@out == null)
            {
                //We allocate a new array
                @out = new NdArray<T>(broadcastshapes.Item1.Plain);
            }
            else
            {
                if (@out.Shape.Dimensions.Length != broadcastshapes.Item1.Dimensions.Length)
                    throw new Exception("Target array does not have the right number of dimensions");

                for (long i = 0; i < @out.Shape.Dimensions.Length; i++)
                    if (@out.Shape.Dimensions[i].Length != broadcastshapes.Item1.Dimensions[i].Length)
                        throw new Exception("Dimension size of target array is incorrect");
            }

            return new Tuple<Shape, Shape, NdArray<T>>(broadcastshapes.Item1, broadcastshapes.Item2, @out);
        }

        /// <summary>
        /// Setup function for shaping the input and output arrays to broadcast compatible shapes.
        /// If no output array is given, a compatible output array is created
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="in1">The input array</param>
        /// <param name="out">The output target</param>
        /// <returns>A compatible output array or throws an exception</returns>
        private static NdArray<T> SetupApplyHelper<T>(NdArray<T> in1, NdArray<T> @out)
        {
            if (@out == null)
            {
                //We allocate a new array
                @out = new NdArray<T>(in1.Shape.Plain);
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
            Tuple<Shape, Shape, NdArray<T>> v = SetupApplyHelper(in1, in2, @out);
            @out = v.Item3;

            if (@out.DataAccessor is ILazyAccessor<T>)
                ((ILazyAccessor<T>)@out.DataAccessor).AddOperation(op, @out, new NdArray<T>(in1, v.Item1), new NdArray<T>(in2, v.Item2));
            else
                FlushMethods.ApplyBinaryOp<T, C>(op, new NdArray<T>(in1, v.Item1), new NdArray<T>(in2, v.Item2), @out);

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
        /// Applies the operation to the input array and scalar value
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="in1">The left-hand-side input argument</param>
        /// <param name="scalar">The right-hand-side scalar value</param>
        /// <param name="out">The output target</param>
        /// <returns>The output value</returns>
        public static NdArray<T> Apply<T, C>(NdArray<T> in1, T scalar, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            return Apply_Entry_Unary<T, ScalarOp<T, C>>(new ScalarOp<T, C>(scalar, new C()), in1, @out);
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
            NdArray<T> v = SetupApplyHelper<T>(in1, @out);

            if (v.DataAccessor is ILazyAccessor<T>)
                ((ILazyAccessor<T>)v.DataAccessor).AddOperation(op, v, in1);
            else
                FlushMethods.ApplyUnaryOp<T, C>(op, in1, v);

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
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        public static NdArray<Tb> Apply<Ta, Tb, C>(NdArray<Ta> in1, NdArray<Tb> @out = null)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            NdArray<Tb> v = SetupApplyHelper(in1, @out);
            UFunc_Op_Inner_UnaryConv<Ta, Tb, C>(in1, v);

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
                FlushMethods.ApplyNullaryOp<T, C>(op, @out);
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