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
        /// The class that serves as an entry point for executing operations
        /// </summary>
        public static class ApplyManager
        {
            /// <summary>
            /// Helper class to sort lists based on priority
            /// </summary>
            private class TupleComparer<T> : IComparer<Tuple<int, T>> 
            {
                public int Compare(Tuple<int, T> x, Tuple<int, T> y)
                {
                    return x.Item1 - y.Item1;
                }
            }
            
            /// <summary>
            /// The priority-sorted list of registered handlers
            /// </summary>
            private static List<Tuple<int, IApplyHandler>> _handlers = new List<Tuple<int, IApplyHandler>>();
            
            /// <summary>
            /// Registers binary operators
            /// </summary>
            /// <param name="handler">The handler to register</param>
            /// <param name="priority">The handler priority</param>
            public static void RegisterHandler(IApplyHandler handler, int priority = 0)
            {
                _handlers.Add(new Tuple<int, IApplyHandler>(priority, handler));
                _handlers.Sort(new TupleComparer<IApplyHandler>());
            }
                    
            /// <summary>
            /// Unregisters a handler.
            /// </summary>
            /// <param name="handler">The handler to unregister</param>
            public static void UnregisterOps(IApplyHandler handler)
            {
                _handlers.RemoveAll(x => x.Item2 == handler);                
            }
        
            /// <summary>
            /// Applies a binary operation using the two operands.
            /// Assumes that the target array is allocated and shaped for broadcast.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="in1">One input operand</param>
            /// <param name="in2">Another input operand</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            /// <returns>True if the operation was applied, false otherwise</returns>            
            public static void ApplyBinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
                where C : struct, IBinaryOp<T>
            {
                foreach (var n in _handlers)
                    if (n.Item2.ApplyBinaryOp<T, C>(op, in1, in2, @out))
                        return;
                
                //Fallback
                Threads.BinaryOp<T, C>(op, in1, in2, @out);
            }

            /// <summary>
            /// Applies a binary operation using the two operands.
            /// Assumes that the target array is allocated and shaped for broadcast.
            /// </summary>
            /// <typeparam name="Ta">The type of input data to operate on</typeparam>
            /// <typeparam name="Tb">The type of output data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="in1">One input operand</param>
            /// <param name="in2">Another input operand</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            /// <returns>True if the operation was applied, false otherwise</returns>            
            public static void ApplyBinaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out)
                where C : struct, IBinaryConvOp<Ta, Tb>
            {
                foreach (var n in _handlers)
                    if (n.Item2.ApplyBinaryConvOp<Ta, Tb, C>(op, in1, in2, @out))
                        return;
                        
                //Fallback
                Threads.BinaryConvOp<Ta, Tb, C>(op, in1, in2, @out);
            }

            /// <summary>
            /// Applies a unary operation using the input operand without lazy evaluation.
            /// Assumes that the target array is allocated and shaped for broadcast.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="in1">The input operand</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            /// <returns>True if the operation was applied, false otherwise</returns>            
            public static void ApplyUnaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> @out)
                where C : struct, IUnaryOp<T>
            {
                foreach (var n in _handlers)
                    if (n.Item2.ApplyUnaryOp<T, C>(op, in1, @out))
                        return;
                       
                //Fallback 
                Threads.UnaryOp<T, C>(op, in1, @out);
            }

            /// <summary>
            /// Applies a unary conversion operation using the input operand without lazy evaluation.
            /// Assumes that the target array is allocated and shaped for broadcast.
            /// </summary>
            /// <typeparam name="Ta">The type of data to convert from</typeparam>
            /// <typeparam name="Tb">The type of data to convert to</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="in1">The input operand</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            /// <returns>True if the operation was applied, false otherwise</returns>            
            public static void ApplyUnaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out)
                where C : struct, IUnaryConvOp<Ta, Tb>
            {
                foreach (var n in _handlers)
                    if (n.Item2.ApplyUnaryConvOp<Ta, Tb, C>(op, in1, @out))
                        return;
                        
                //Fallback 
                Threads.UnaryConvOp<Ta, Tb, C>(op, in1, @out);
            }

            /// <summary>
            /// Applies a nullary operation to each element in the output operand without lazy evaluation.
            /// Assumes that the target array is allocated.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to perform</typeparam>
            /// <param name="op">The operation to use</param>
            /// <param name="out">The target operand, must be allocated and shaped for broadcast</param>
            /// <returns>True if the operation was applied, false otherwise</returns>            
            public static void ApplyNullaryOp<T, C>(C op, NdArray<T> @out)
                where C : struct, INullaryOp<T>
            {
                foreach (var n in _handlers)
                    if (n.Item2.ApplyNullaryOp<T, C>(op, @out))
                        return;
                        
                //Fallback 
                Threads.NullaryOp<T, C>(op, @out);
            }

            /// <summary>
            /// Reduces the input argument on the specified axis without lazy evaluation.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="C">The type of operation to reduce with</typeparam>
            /// <param name="op">The operation to perform</param>
            /// <param name="in1">The input argument</param>
            /// <param name="axis">The axis to reduce</param>
            /// <param name="out">The output target</param>
            /// <returns>True if the operation was applied, false otherwise</returns>            
            public static void ApplyReduce<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out)
                where C : struct, IBinaryOp<T>
            {
                foreach (var n in _handlers)
                    if (n.Item2.ApplyReduce<T, C>(op, axis, in1, @out))
                        return;
                        
                //Fallback 
                Threads.Reduce<T, C>(op, axis, in1, @out);
            }

            /// <summary>
            /// Performs matrix multiplication on the two operands, using the supplied methods,
            /// without using lazy evaluation.
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <typeparam name="CADD">The typed add operator</typeparam>
            /// <typeparam name="CMUL">The typed multiply operator</typeparam>
            /// <param name="addop">The add operator</param>
            /// <param name="mulop">The multiply operator</param>
            /// <param name="in1">The left-hand-side argument</param>
            /// <param name="in2">The right-hand-side argument</param>
            /// <param name="out">An optional output argument, use for in-place operations</param>
            /// <returns>True if the operation was applied, false otherwise</returns>            
            public static void ApplyMatmul<T, CADD, CMUL>(CADD addop, CMUL mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null)
                where CADD : struct, IBinaryOp<T>
                where CMUL : struct, IBinaryOp<T>
            {
                foreach (var n in _handlers)
                    if (n.Item2.ApplyMatmul<T, CADD, CMUL>(addop, mulop, in1, in2, @out))
                        return;

                //Fallback 
                UFunc.UFunc_Matmul_Inner_Flush<T, CADD, CMUL>(addop, mulop, in1, in2, @out);
            }

            /// <summary>
            /// Calculates the scalar result of applying the binary operation to all elements
            /// </summary>
            /// <typeparam name="T">The value to operate on</typeparam>
            /// <typeparam name="C">The operation to perform</typeparam>
            /// <param name="op">The operation to reduce with</param>
            /// <param name="in1">The array to aggregate</param>
            /// <param name="result">A scalar value that is the result of aggregating all elements</param>
            /// <returns>True if the operation was applied, false otherwise</returns>            
            public static void ApplyAggregate<T, C>(C op, NdArray<T> in1, out T result)
                where C : struct, IBinaryOp<T>
            {
                foreach (var n in _handlers)
                    if (n.Item2.ApplyAggregate<T, C>(op, in1, out result))
                        return;
                                                
                //Fallback 
                result = UFunc.UFunc_Aggregate_Inner_Flush<T, C>(op, in1);
            }        
        }
    }    
}

