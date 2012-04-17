using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL
{
    public partial class UFunc
    {
        public struct LazyReduceOperation<T> : IBinaryOp<T>
        {
            public readonly long Axis;
            public readonly IBinaryOp<T> Operation;
            public LazyReduceOperation(IBinaryOp<T> operation, long axis) 
            {
                Operation = operation;
                Axis = axis; 
            }

            public T Op(T a, T b)
            {
                throw new NotImplementedException();
            }
        }

        public static NdArray<T> SetupReduceHelper<T>(NdArray<T> in1, long axis, NdArray<T> @out)
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

        public static NdArray<T> Reduce<T, C>(NdArray<T> in1, long axis = 0, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            return Reduce_Entry<T, C>(new C(), in1, axis, @out);
        }

        private static NdArray<T> Reduce_Entry<T, C>(C op, NdArray<T> in1, long axis = 0, NdArray<T> @out = null)
            where C : struct, IBinaryOp<T>
        {
            NdArray<T> v = SetupReduceHelper<T>(in1, axis, @out);

            if (v.m_data is ILazyAccessor<T>)
                ((ILazyAccessor<T>)v.m_data).AddOperation(new LazyReduceOperation<T>(new C(), axis), v, in1);
            else
                return UFunc_Reduce_Inner_Flush<T, C>(op, axis, in1, v);

            return v;
        }

        public static NdArray<T> Reduce<T>(IBinaryOp<T> op, NdArray<T> in1, long axis = 0, NdArray<T> @out = null)
        {
            var method = typeof(UFunc).GetMethod("Reduce_Entry", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
            var gm = method.MakeGenericMethod(typeof(T), op.GetType());
            return (NdArray<T>)gm.Invoke(null, new object[] { op, in1, axis, @out });
        }

        private static NdArray<T> UFunc_Reduce_Inner_Flush<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                UFunc_Op_Inner_Unary_Flush<T, CopyOp<T>>(new CopyOp<T>(), new NdArray<T>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                T[] d = in1.Data;
                T[] vd = @out.Data;

                //Simple case, reduce 1D array to scalar value
                if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                {
                    long stride = in1.Shape.Dimensions[0].Stride;
                    long ix = in1.Shape.Offset;
                    long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                    T value = op.Op(d[ix], d[ix + stride]);

                    for (long i = ix + (stride * 2); i < ix + limit; i += stride)
                        value = op.Op(value, d[i]);

                    vd[@out.Shape.Offset] = value;
                }
                //Simple case, reduce 2D array to 1D
                else if (axis == 0 && in1.Shape.Dimensions.LongLength == 2)
                {
                    long strideInner = in1.Shape.Dimensions[1].Stride;
                    long strideOuter = in1.Shape.Dimensions[0].Stride;

                    long ix = in1.Shape.Offset;
                    long limitInner = strideInner * in1.Shape.Dimensions[1].Length;

                    long ox = @out.Shape.Offset;
                    long strideRes = @out.Shape.Dimensions[0].Stride;

                    long outerCount = in1.Shape.Dimensions[0].Length;

                    for (long i = 0; i < in1.Shape.Dimensions[1].Length; i++)
                    {
                        T value = d[ix];

                        long nx = ix;
                        for (long j = 1; j < outerCount; j++)
                        {
                            nx += strideOuter;
                            value = op.Op(value, d[nx]);
                        }

                        vd[ox] = value;
                        ox += strideRes;

                        ix += strideInner;
                    }
                }
                //Simple case, reduce 2D array to 1D
                else if (axis == 1 && in1.Shape.Dimensions.LongLength == 2)
                {
                    long strideInner = in1.Shape.Dimensions[1].Stride;
                    long strideOuter = in1.Shape.Dimensions[0].Stride;

                    long ix = in1.Shape.Offset;
                    long limitInner = strideInner * in1.Shape.Dimensions[1].Length;

                    long ox = @out.Shape.Offset;
                    long strideRes = @out.Shape.Dimensions[0].Stride;

                    for (long i = 0; i < in1.Shape.Dimensions[0].Length; i++)
                    {
                        T value = d[ix];

                        for (long j = strideInner; j < limitInner; j += strideInner)
                            value = op.Op(value, d[j + ix]);

                        vd[ox] = value;
                        ox += strideRes;

                        ix += strideOuter;
                    }
                }                
                //General case
                else
                {
                    long size = in1.Shape.Dimensions[axis].Length;
                    Range[] inRanges = new Range[in1.Shape.Dimensions.LongLength];
                    Range[] outRanges = new Range[in1.Shape.Dimensions.LongLength];
                    
                    //We make a fake shape for the arrays, so they mach
                    for (long i = 0; i < inRanges.LongLength; i++)
                        inRanges[i] = i == axis ? Range.El(0) : Range.All;
                    for (long i = 0; i < outRanges.LongLength; i++)
                        outRanges[i] = i == axis ? Range.NewAxis : Range.All;

                    NdArray<T> vl = @out[outRanges];

                    //Initially we just copy the value
                    UFunc_Op_Inner_Unary_Flush<T, CopyOp<T>>(new CopyOp<T>(), in1[inRanges], ref vl);
                    
                    //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                    for (long j = 1; j < size; j++)
                    {
                        //Select the new dimension
                        inRanges[axis] = Range.El(j);
                        //Apply the operation
                        UFunc_Op_Inner_Binary_Flush<T, C>(op, vl, in1[inRanges], ref vl);
                    }
                }
            }
            return @out;
        }

    }
}
