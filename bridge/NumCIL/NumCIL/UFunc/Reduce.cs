using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL
{
    public partial class UFunc
    {
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
            NdArray<T> v = SetupReduceHelper<T>(in1, axis, @out);

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                UFunc_Op_Inner<T, CopyOp<T>>(new NdArray<T>(in1, new Shape(sizes, in1.Shape.Offset)), ref v);
            }
            else
            {
                C op = new C();
                T[] d = in1.Data;
                T[] vd = v.Data;

                long size = in1.Shape.Dimensions[axis].Length;

                long[] limits = new long[axis];
                long[] counters = new long[axis];
                Array.Copy(
                    in1.Shape.Dimensions.Select(x => x.Length).ToArray(),
                    0,
                    limits,
                    0,
                    axis);

                if (limits.LongLength == 0 && in1.Shape.Dimensions.LongLength == 2)
                {
                    long strideInner = in1.Shape.Dimensions[1].Stride;
                    long strideOuter = in1.Shape.Dimensions[0].Stride;

                    long ix = in1.Shape.Offset;
                    long limitInner = strideInner * in1.Shape.Dimensions[1].Length;

                    long ox = v.Shape.Offset;
                    long strideRes = v.Shape.Dimensions[0].Stride;

                    for (long i = 0; i < in1.Shape.Dimensions[0].Length; i++)
                    {
                        T value = op.Op(d[ix], d[ix+strideInner]);

                        long lm = ix + limitInner;

                        for (long j = ix + (strideInner * 2); j < lm; j += strideInner)
                            value = op.Op(value, d[j]);

                        vd[ox] = value;
                        ox += strideRes;

                        ix += strideOuter;
                    }
                }
                else if (limits.LongLength == 0 && in1.Shape.Dimensions.LongLength == 1)
                {
                    long stride = in1.Shape.Dimensions[0].Stride;
                    long ix = in1.Shape.Offset;
                    long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                    T value = op.Op(d[ix], d[ix + stride]);

                    for (long i = ix + (stride * 2); i < ix + limit; i += stride)
                        value = op.Op(value, d[i]);

                    vd[v.Shape.Offset] = value;
                }
                else
                {
                    long totalOps = limits.Aggregate<long>((a, b) => a * b);

                    for (long i = 0; i < totalOps; i++)
                    {
                        NdArray<T> vl = v[counters];

                        NdArray<T> in1V = in1[counters];

                        long offset = in1V.Shape.Offset;
                        long stride = in1V.Shape.Dimensions[0].Stride;

                        if (in1V.Shape.Dimensions.LongLength == 1)
                        {
                            long pos = offset + stride;
                            T a = op.Op(d[offset], d[pos]);
                            for (long j = 2; j < size; j++)
                            {
                                pos += stride;
                                a = op.Op(a, d[pos]);
                            }

                            vl.Value[0] = a;
                        }
                        else
                        {
                            UFunc_Op_Inner<T, C>(in1V[0], in1V[1], ref vl);

                            for (long j = 2; j < size; j++)
                                UFunc_Op_Inner<T, C>(vl, in1V[j], ref vl);
                        }

                        //Basically a ripple carry adder
                        long p = counters.LongLength - 1;
                        while (totalOps > 1 && ++counters[p] == limits[p] && p > 0)
                        {
                            counters[p] = 0;
                            p--;
                        }
                    }
                }
                
            }
            return v;
        }

    }
}
