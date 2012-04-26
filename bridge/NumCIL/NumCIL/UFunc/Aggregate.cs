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
            //TODO: This can be implemented without reduce, and would be faster and use less memory
            while (in1.Shape.Dimensions.LongLength > 1)
                in1 = UFunc.Reduce_Entry<T, C>(op, in1, 0);

            return UFunc.Reduce_Entry<T, C>(op, in1, 0).Value[0];

        }
    }
}
