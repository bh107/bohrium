#region Copyright
/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;
using System.Threading;

namespace NumCIL
{
    public static partial class UFunc
    {
        /// <summary>
        /// Threaded implementation of various ufunc apply methods
        /// </summary>
        public static class Threads
        {
            /// <summary>
            /// A value controlling if single workblock should use a single thread or run directly
            /// </summary>
            private const bool SINGLE_CORE_THREAD = false;

            /// <summary>
            /// The number of work blocks
            /// </summary>
            private static int _no_blocks = Math.Max(2, Environment.ProcessorCount * 2);

            /// <summary>
            /// Static constructor, used to extract blocksize from environment variable
            /// </summary>
            static Threads()
            {
                int p;
                if (int.TryParse(Environment.GetEnvironmentVariable("NUMCIL_BLOCKSIZE"), out p))
                    BlockCount = p;
            }

            /// <summary>
            /// Gets or sets the number of work blocks processed by threads
            /// </summary>
            public static int BlockCount { get { return _no_blocks; } set { _no_blocks = Math.Max(1, value); } }

            /// <summary>
            /// Reshapes an array suitable for parallel evaluation
            /// </summary>
            /// <typeparam name="T">The type of data to operate on</typeparam>
            /// <param name="arg">The argument to reshape</param>
            /// <param name="blockno">The index of this block</param>
            /// <param name="nblocks">The total number of blocks</param>
            /// <param name="dimension">The dimension to reshape</param>
            /// <returns>A reshaped array</returns>
            private static NdArray<T> Reshape<T>(NdArray<T> arg, int blockno, int nblocks, long dimension = 0)
            {
                //This reshapes over dimension zero, and does not really work if the array has an extra first dimension
                long blockoffset = blockno == 0 ? 0 :
                    ((arg.Shape.Dimensions[dimension].Length / nblocks) * blockno) +
                    arg.Shape.Dimensions[dimension].Length % nblocks;

                long[] lengths = arg.Shape.Dimensions.Select(x => x.Length).ToArray();
                long offset = arg.Shape.Offset + (blockoffset * arg.Shape.Dimensions[dimension].Stride);
                lengths[dimension] = (arg.Shape.Dimensions[dimension].Length / nblocks) + (blockno == 0 ? arg.Shape.Dimensions[dimension].Length % nblocks : 0);

                return new NdArray<T>(arg.DataAccessor, new Shape(lengths, offset, arg.Shape.Dimensions.Select(x => x.Stride).ToArray()));
            }

            internal static void BinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
                where C : struct, IBinaryOp<T>
            {
                if ((SINGLE_CORE_THREAD || _no_blocks > 1) && @out.Shape.Dimensions[0].Length >= _no_blocks)
                {
                    AutoResetEvent cond = new AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    in1.DataAccessor.Allocate();
                    in2.DataAccessor.Allocate();
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        NumCIL.ThreadPool.QueueUserWorkItem((args) => 
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_Binary_Flush(op, Reshape(in1, block, totalblocks), Reshape(in2, block, totalblocks), Reshape(@out, block, totalblocks));

                            Interlocked.Increment(ref blocksdone);
                            cond.Set();    
                        }, i);
                    }

                    while (blocksdone < totalblocks) 
                        cond.WaitOne();
                }
                else
                {
                    UFunc.UFunc_Op_Inner_Binary_Flush(op, in1, in2, @out);
                }
            }

            internal static void BinaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out)
                where C : struct, IBinaryConvOp<Ta, Tb>
            {
                if ((SINGLE_CORE_THREAD || _no_blocks > 1) && @out.Shape.Dimensions[0].Length >= _no_blocks)
                {
                    AutoResetEvent cond = new AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    in1.DataAccessor.Allocate();
                    in2.DataAccessor.Allocate();
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        NumCIL.ThreadPool.QueueUserWorkItem((args) =>
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_BinaryConv_Flush(op, Reshape(in1, block, totalblocks), Reshape(in2, block, totalblocks), Reshape(@out, block, totalblocks));

                            Interlocked.Increment(ref blocksdone);
                            cond.Set();
                        }, i);
                    }

                    while (blocksdone < totalblocks)
                        cond.WaitOne();
                }
                else
                {
                    UFunc.UFunc_Op_Inner_BinaryConv_Flush(op, in1, in2, @out);
                }
            }
            internal static void UnaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> @out)
                where C : struct, IUnaryOp<T>
            {
                if ((SINGLE_CORE_THREAD || _no_blocks > 1) && @out.Shape.Dimensions[0].Length >= _no_blocks)
                {
                    AutoResetEvent cond = new AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    in1.DataAccessor.Allocate();
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        NumCIL.ThreadPool.QueueUserWorkItem((args) =>
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_Unary_Flush(op, Reshape(in1, block, totalblocks), Reshape(@out, block, totalblocks));

                            Interlocked.Increment(ref blocksdone);
                            cond.Set();
                        }, i);
                    }

                    while (blocksdone < totalblocks)
                        cond.WaitOne();
                }
                else
                {
                    UFunc.UFunc_Op_Inner_Unary_Flush(op, in1, @out);
                }

            }

            internal static void Reduce<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out)
                where C : struct, IBinaryOp<T>
            {
				bool largeInput = in1.Shape.Dimensions[0].Length >= _no_blocks;
				bool largeOutput = @out.Shape.Dimensions[0].Length >= _no_blocks && @out.Shape.Dimensions[Math.Max(0, axis-1)].Length >= _no_blocks;
				bool scalarReduction = axis == 0 && in1.Shape.Dimensions.LongLength == 1;
				bool doubleLargeInput = in1.Shape.Dimensions[0].Length >= (_no_blocks * 2);

                if ((SINGLE_CORE_THREAD || _no_blocks > 1) && ((largeInput && largeOutput) || (doubleLargeInput && scalarReduction)))
                {
                    AutoResetEvent cond = new AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    in1.DataAccessor.Allocate();
                    @out.DataAccessor.Allocate();

					//Special handling required for 1D to scalar
	                if (axis == 0 && in1.Shape.Dimensions.LongLength == 1)
					{
						//Allocate some temp storage
						T[] tmpout = new T[totalblocks];
	                    for (int i = 0; i < totalblocks; i++)
	                    {
	                        NumCIL.ThreadPool.QueueUserWorkItem((args) =>
	                        {
	                            int block = (int)args;

								UFunc.UFunc_Reduce_Inner_Flush(op, axis, Reshape(in1, block, totalblocks), new NdArray<T>(tmpout, new Shape(new long[] { 1 }, block)));

	                            Interlocked.Increment(ref blocksdone);
	                            cond.Set();
	                        }, i);
	                    }

	                    while (blocksdone < totalblocks)
	                        cond.WaitOne();

						//Make the final reduction on the thread results
						T r = tmpout[0];
						for(int i = 1; i < totalblocks; i++)
							r = op.Op(r, tmpout[i]);

						@out.Value[@out.Shape.Offset] = r;
					}
					else
					{
	                    for (int i = 0; i < totalblocks; i++)
	                    {
	                        NumCIL.ThreadPool.QueueUserWorkItem((args) =>
	                        {
	                            int block = (int)args;
								var v1 = Reshape(in1, block, totalblocks, axis == 0 ? 1 : axis - 1);
								var v2 = Reshape(@out, block, totalblocks, axis == 0 ? 0 : axis - 1);
	                            UFunc.UFunc_Reduce_Inner_Flush(op, axis, v1, v2);

	                            Interlocked.Increment(ref blocksdone);
	                            cond.Set();
	                        }, i);
	                    }

	                    while (blocksdone < totalblocks)
	                        cond.WaitOne();
					}
                }
                else
                {
                    UFunc.UFunc_Reduce_Inner_Flush(op, axis, in1, @out);
                }

            }

            internal static void NullaryOp<T, C>(C op, NdArray<T> @out)
                where C : struct, INullaryOp<T>
            {
                if ((SINGLE_CORE_THREAD || _no_blocks > 1) && @out.Shape.Dimensions[0].Length >= _no_blocks)
                {
                    AutoResetEvent cond = new AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        NumCIL.ThreadPool.QueueUserWorkItem((args) =>
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_Nullary_Flush(op, Reshape(@out, block, totalblocks));

                            Interlocked.Increment(ref blocksdone);
                            cond.Set();
                        }, i);
                    }

                    while (blocksdone < totalblocks)
                        cond.WaitOne();
                }
                else
                {
                    UFunc.UFunc_Op_Inner_Nullary_Flush(op, @out);
                }
            }

            internal static void UnaryConvOp<Ta, Tb, C>(IUnaryConvOp<Ta, Tb> op, NdArray<Ta> in1, NdArray<Tb> @out)
                where C : struct, IUnaryConvOp<Ta, Tb>
            {
                if ((SINGLE_CORE_THREAD || _no_blocks > 1) && @out.Shape.Dimensions[0].Length >= _no_blocks)
                {
                    AutoResetEvent cond = new AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    in1.DataAccessor.Allocate();
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        NumCIL.ThreadPool.QueueUserWorkItem((args) =>
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_UnaryConv_Flush(op, Reshape(in1, block, totalblocks), Reshape(@out, block, totalblocks));

                            Interlocked.Increment(ref blocksdone);
                            cond.Set();
                        }, i);
                    }

                    while (blocksdone < totalblocks)
                        cond.WaitOne();
                }
                else
                {
                    UFunc.UFunc_Op_Inner_UnaryConv_Flush(op, in1, @out);
                }
            }
        }
    }
}
