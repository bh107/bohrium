using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL
{
    //Support for .Net 3.5
    public class Tuple<T1>
    {
        public T1 Item1;
        public Tuple(T1 t1) { Item1 = t1; }
    }

    public class Tuple<T1, T2> : Tuple<T1> 
    { 
        public T2 Item2;
        
        public Tuple(T1 t1, T2 t2)
            : base(t1)
        { Item2 = t2; }
    }

    public class Tuple<T1, T2, T3> : Tuple<T1, T2>
    {
        public T3 Item3;

        public Tuple(T1 t1, T2 t2, T3 t3)
            : base(t1, t2)
        { Item3 = t3; }
    }

    public partial class UFunc
    {
        public static Tuple<Shape, Shape, NdArray<T>> SetupApplyHelper<T>(NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
        {
            Tuple<Shape, Shape> broadcastshapes = Shape.ToBroadcastShapes(in1.Shape, in2.Shape);
            if (@out == null)
            {
                //We allocate a new array
                @out = new NdArray<T>(new Shape(broadcastshapes.Item1.Dimensions.Select(x => x.Length).ToArray()));
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

        public static NdArray<T> SetupApplyHelper<T>(NdArray<T> in1, NdArray<T> @out)
        {
            if (@out == null)
            {
                //We allocate a new array
                @out = new NdArray<T>(in1.Shape);
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

        public static NdArray<Tb> SetupApplyHelper<Ta, Tb>(NdArray<Ta> in1, NdArray<Tb> @out)
        {
            if (@out == null)
            {
                //We allocate a new array
                @out = new NdArray<Tb>(in1.Shape);
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

        public static NdArray<T> Apply<T, C>(NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            
            Tuple<Shape, Shape, NdArray<T>> v = SetupApplyHelper(in1, in2, @out);
            @out = v.Item3;
            
            UFunc_Op_Inner<T, C>(new NdArray<T>(in1, v.Item1), new NdArray<T>(in2, v.Item2), ref @out);

            return @out;
        }

        public static NdArray<T> Apply<T, C>(NdArray<T> in1, T scalar, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            @out = SetupApplyHelper<T>(in1, @out);

            UFunc_Op_Inner<T, C>(in1, scalar, ref @out);

            return @out;
        }

        public static NdArray<T> Apply<T, C>(NdArray<T> in1, NdArray<T> @out = null)
            where C : struct, IUnaryOp<T>
        {
            NdArray<T> v = SetupApplyHelper<T>(in1, @out);
            UFunc_Op_Inner<T, C>(in1, ref v);

            return v;
        }

        public static NdArray<Tb> Apply<Ta, Tb, C>(NdArray<Ta> in1, NdArray<Tb> @out = null)
            where C : struct, IUnaryConvOp<Ta, Tb>
        {
            NdArray<Tb> v = SetupApplyHelper(in1, @out);
            UFunc_Op_Inner<Ta, Tb, C>(in1, ref v);

            return v;
        }

        public static NdArray<T> Apply<T>(Func<T, T, T> op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
        {
            Tuple<Shape, Shape, NdArray<T>> v = SetupApplyHelper(in1, in2, @out);
            @out = v.Item3;

            UFunc_Op_Inner<T, BinaryLambdaOp<T>>(op, new NdArray<T>(in1, v.Item1), new NdArray<T>(in2, v.Item2), ref @out);

            return v.Item3;
        }

        public static NdArray<T> Apply<T>(Func<T, T> op, NdArray<T> in1, NdArray<T> @out = null)
        {
            NdArray<T> v = SetupApplyHelper<T>(in1, @out);

            UFunc_Op_Inner<T, UnaryLambdaOp<T>>(op, in1, ref @out);

            return v;
        }

        public static void Apply<T, C>(C op, NdArray<T> @out)
            where C : struct, INullaryOp<T>
        {
            UFunc_Op_Inner<T, C>(op, @out);
        }
    }
}