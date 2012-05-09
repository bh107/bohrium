using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL.Unsafe
{
    internal static class Reduce
    {
        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<float> UFunc_Reduce_Inner_Flush_Single<C>(C op, long axis, NdArray<float> in1, NdArray<float> @out)
            where C : struct, IBinaryOp<float>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_Single<CopyOp<float>>(new CopyOp<float>(), new NdArray<float>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (float* d = in1.Data)
                    fixed (float* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            float value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                float value = d[ix];

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
                                float value = d[ix];

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
                            NdArray<float> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_Single<CopyOp<float>>(new CopyOp<float>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_Single<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<double> UFunc_Reduce_Inner_Flush_Double<C>(C op, long axis, NdArray<double> in1, NdArray<double> @out)
            where C : struct, IBinaryOp<double>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_Double<CopyOp<double>>(new CopyOp<double>(), new NdArray<double>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (double* d = in1.Data)
                    fixed (double* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            double value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                double value = d[ix];

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
                                double value = d[ix];

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
                            NdArray<double> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_Double<CopyOp<double>>(new CopyOp<double>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_Double<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<long> UFunc_Reduce_Inner_Flush_Int64<C>(C op, long axis, NdArray<long> in1, NdArray<long> @out)
            where C : struct, IBinaryOp<long>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_Int64<CopyOp<long>>(new CopyOp<long>(), new NdArray<long>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (long* d = in1.Data)
                    fixed (long* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            long value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                long value = d[ix];

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
                                long value = d[ix];

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
                            NdArray<long> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_Int64<CopyOp<long>>(new CopyOp<long>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_Int64<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<ulong> UFunc_Reduce_Inner_Flush_UInt64<C>(C op, long axis, NdArray<ulong> in1, NdArray<ulong> @out)
            where C : struct, IBinaryOp<ulong>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_UInt64<CopyOp<ulong>>(new CopyOp<ulong>(), new NdArray<ulong>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (ulong* d = in1.Data)
                    fixed (ulong* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            ulong value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                ulong value = d[ix];

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
                                ulong value = d[ix];

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
                            NdArray<ulong> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_UInt64<CopyOp<ulong>>(new CopyOp<ulong>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_UInt64<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<int> UFunc_Reduce_Inner_Flush_Int32<C>(C op, long axis, NdArray<int> in1, NdArray<int> @out)
            where C : struct, IBinaryOp<int>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_Int32<CopyOp<int>>(new CopyOp<int>(), new NdArray<int>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (int* d = in1.Data)
                    fixed (int* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            int value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                int value = d[ix];

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
                                int value = d[ix];

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
                            NdArray<int> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_Int32<CopyOp<int>>(new CopyOp<int>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_Int32<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<uint> UFunc_Reduce_Inner_Flush_UInt32<C>(C op, long axis, NdArray<uint> in1, NdArray<uint> @out)
            where C : struct, IBinaryOp<uint>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_UInt32<CopyOp<uint>>(new CopyOp<uint>(), new NdArray<uint>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (uint* d = in1.Data)
                    fixed (uint* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            uint value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                uint value = d[ix];

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
                                uint value = d[ix];

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
                            NdArray<uint> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_UInt32<CopyOp<uint>>(new CopyOp<uint>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_UInt32<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<short> UFunc_Reduce_Inner_Flush_Int16<C>(C op, long axis, NdArray<short> in1, NdArray<short> @out)
            where C : struct, IBinaryOp<short>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_Int16<CopyOp<short>>(new CopyOp<short>(), new NdArray<short>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (short* d = in1.Data)
                    fixed (short* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            short value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                short value = d[ix];

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
                                short value = d[ix];

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
                            NdArray<short> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_Int16<CopyOp<short>>(new CopyOp<short>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_Int16<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<ushort> UFunc_Reduce_Inner_Flush_UInt16<C>(C op, long axis, NdArray<ushort> in1, NdArray<ushort> @out)
            where C : struct, IBinaryOp<ushort>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_UInt16<CopyOp<ushort>>(new CopyOp<ushort>(), new NdArray<ushort>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (ushort* d = in1.Data)
                    fixed (ushort* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            ushort value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                ushort value = d[ix];

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
                                ushort value = d[ix];

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
                            NdArray<ushort> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_UInt16<CopyOp<ushort>>(new CopyOp<ushort>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_UInt16<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<byte> UFunc_Reduce_Inner_Flush_Byte<C>(C op, long axis, NdArray<byte> in1, NdArray<byte> @out)
            where C : struct, IBinaryOp<byte>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_Byte<CopyOp<byte>>(new CopyOp<byte>(), new NdArray<byte>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (byte* d = in1.Data)
                    fixed (byte* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            byte value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                byte value = d[ix];

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
                                byte value = d[ix];

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
                            NdArray<byte> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_Byte<CopyOp<byte>>(new CopyOp<byte>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_Byte<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }

        /// <summary>
        /// Unsafe implementation of the reduce operation
        /// </summary>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<sbyte> UFunc_Reduce_Inner_Flush_SByte<C>(C op, long axis, NdArray<sbyte> in1, NdArray<sbyte> @out)
            where C : struct, IBinaryOp<sbyte>
        {
            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                Apply.UFunc_Op_Inner_Unary_Flush_SByte<CopyOp<sbyte>>(new CopyOp<sbyte>(), new NdArray<sbyte>(in1, new Shape(sizes, in1.Shape.Offset)), ref @out);
            }
            else
            {
                unsafe
                {
                    fixed (sbyte* d = in1.Data)
                    fixed (sbyte* vd = @out.Data)
                    {

                        //Simple case, reduce 1D array to scalar value
                        if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                        {
                            long stride = in1.Shape.Dimensions[0].Stride;
                            long ix = in1.Shape.Offset;
                            long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                            sbyte value = d[ix];

                            for (long i = ix + stride; i < ix + limit; i += stride)
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
                                sbyte value = d[ix];

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
                                sbyte value = d[ix];

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
                            NdArray<sbyte> vl = @out.Subview(Range.NewAxis, axis);

                            //Initially we just copy the value
                            Apply.UFunc_Op_Inner_Unary_Flush_SByte<CopyOp<sbyte>>(new CopyOp<sbyte>(), in1.Subview(Range.El(0), axis), ref vl);

                            //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                            for (long j = 1; j < size; j++)
                            {
                                //Select the new dimension
                                //Apply the operation
                                Apply.UFunc_Op_Inner_Binary_Flush_SByte<C>(op, vl, in1.Subview(Range.El(j), axis), ref vl);
                            }
                        }
                    }
                }
            }
            return @out;
        }
    }
}
