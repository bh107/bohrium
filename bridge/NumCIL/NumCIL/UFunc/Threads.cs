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
        /// Threaded implementation of various ufunc apply methods
        /// </summary>
        public static class Threads
        {
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
            /// <returns>A reshaped array</returns>
            private static NdArray<T> Reshape<T>(NdArray<T> arg, int blockno, int nblocks)
            {
                //This reshapes over dimension zero, and does not really work if the array has an extra first dimension
                long blockoffset = blockno == 0 ? 0 :
                    ((arg.Shape.Dimensions[0].Length / nblocks) * blockno) +
                    arg.Shape.Dimensions[0].Length % nblocks;

                long[] lengths = arg.Shape.Dimensions.Select(x => x.Length).ToArray();
                long offset = arg.Shape.Offset + (blockoffset * arg.Shape.Dimensions[0].Stride);
                lengths[0] = (arg.Shape.Dimensions[0].Length / nblocks) + (blockno == 0 ? arg.Shape.Dimensions[0].Length % nblocks : 0);

                return new NdArray<T>(arg.DataAccessor, new Shape(lengths, offset, arg.Shape.Dimensions.Select(x => x.Stride).ToArray()));
            }

            internal static void BinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out)
                where C : struct, IBinaryOp<T>
            {
                if (_no_blocks > 1 && @out.Shape.Dimensions[0].Length >= _no_blocks)
                {
                    System.Threading.AutoResetEvent cond = new System.Threading.AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        System.Threading.ThreadPool.QueueUserWorkItem((args) => 
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_Binary_Flush(op, Reshape(in1, block, totalblocks), Reshape(in2, block, totalblocks), Reshape(@out, block, totalblocks));

                            System.Threading.Interlocked.Increment(ref blocksdone);
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

            internal static void UnaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> @out)
                where C : struct, IUnaryOp<T>
            {
                if (_no_blocks > 1 && @out.Shape.Dimensions[0].Length >= _no_blocks && false)
                {
                    System.Threading.AutoResetEvent cond = new System.Threading.AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        System.Threading.ThreadPool.QueueUserWorkItem((args) =>
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_Unary_Flush(op, Reshape(in1, block, totalblocks), Reshape(@out, block, totalblocks));

                            System.Threading.Interlocked.Increment(ref blocksdone);
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

            internal static void NullaryOp<T, C>(C op, NdArray<T> @out)
                where C : struct, INullaryOp<T>
            {
                if (_no_blocks > 1 && @out.Shape.Dimensions[0].Length >= _no_blocks)
                {
                    System.Threading.AutoResetEvent cond = new System.Threading.AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        System.Threading.ThreadPool.QueueUserWorkItem((args) =>
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_Nullary_Flush(op, Reshape(@out, block, totalblocks));

                            System.Threading.Interlocked.Increment(ref blocksdone);
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
                if (_no_blocks > 1 && @out.Shape.Dimensions[0].Length >= _no_blocks && (@out.Shape.Elements / _no_blocks) > 128)
                {
                    System.Threading.AutoResetEvent cond = new System.Threading.AutoResetEvent(false);
                    int blocksdone = 0;
                    int totalblocks = _no_blocks;
                    @out.DataAccessor.Allocate();

                    for (int i = 0; i < totalblocks; i++)
                    {
                        System.Threading.ThreadPool.QueueUserWorkItem((args) =>
                        {
                            int block = (int)args;

                            UFunc.UFunc_Op_Inner_UnaryConv_Flush(op, Reshape(in1, block, totalblocks), Reshape(@out, block, totalblocks));

                            System.Threading.Interlocked.Increment(ref blocksdone);
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
