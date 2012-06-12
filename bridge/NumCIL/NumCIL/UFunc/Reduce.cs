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
                return FlushMethods.Reduce<T, C>(op, axis, in1, v);

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
        /// Actually executes a reduce operation in CIL by retrieving the data and executing the <see cref="T:NumCIL.IBinaryOp{0}"/> on each element in the given dimension.
        /// This implementation is optimized for use with up to 2 dimensions, but works for any size dimension.
        /// This method is optimized for 64bit processors, using the .Net 4.0 runtime.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to reduce with</typeparam>
        /// <param name="op">The instance of the operation to reduce with</param>
        /// <param name="in1">The input argument</param>
        /// <param name="axis">The axis to reduce</param>
        /// <param name="out">The output target</param>
        /// <returns>The output target</returns>
        private static NdArray<T> UFunc_Reduce_Inner_Flush<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (UnsafeAPI.UFunc_Reduce_Inner_Flush_Unsafe<T, C>(op, axis, in1, @out))
                return @out;

            if (axis < 0)
                axis = in1.Shape.Dimensions.LongLength - axis;

            //Basic case, just return a reduced array
            if (in1.Shape.Dimensions[axis].Length == 1 && in1.Shape.Dimensions.LongLength > 1)
            {
                //TODO: If both in and out use the same array, just return a reshaped in
                long j = 0;
                var sizes = in1.Shape.Dimensions.Where(x => j++ != axis).ToArray();
                UFunc_Op_Inner_Unary_Flush<T, CopyOp<T>>(new CopyOp<T>(), new NdArray<T>(in1, new Shape(sizes, in1.Shape.Offset)), @out);
            }
            else
            {
                T[] d = in1.AsArray();
                T[] vd = @out.AsArray();

                //Simple case, reduce 1D array to scalar value
                if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
                {
                    long stride = in1.Shape.Dimensions[0].Stride;
                    long ix = in1.Shape.Offset;
                    long limit = (stride * in1.Shape.Dimensions[0].Length) + ix;

                    T value = d[ix];

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
                    NdArray<T> vl = @out.Subview(Range.NewAxis, axis);

                    //Initially we just copy the value
                    FlushMethods.ApplyUnaryOp<T, CopyOp<T>>(new CopyOp<T>(), in1.Subview(Range.El(0), axis), vl);
                    
                    //If there is more than one element in the dimension to reduce, apply the operation accumulatively
                    for (long j = 1; j < size; j++)
                    {
                        //Select the new dimension
                        //Apply the operation
                        FlushMethods.ApplyBinaryOp<T, C>(op, vl, in1.Subview(Range.El(j), axis), vl);
                    }
                }
            }
            return @out;
        }

    }
}
