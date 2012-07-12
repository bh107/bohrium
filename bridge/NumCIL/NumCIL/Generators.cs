using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.Generic
{
    /// <summary>
    /// Interface for implementing a custom generator
    /// </summary>
    /// <typeparam name="T">The type of data to generate</typeparam>
    public interface IGenerator<T>
    {
        /// <summary>
        /// Generates an NdArray with sequential integers, starting with zero
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with sequential integers, starting with zero</returns>
        NdArray<T> Arange(long size);
        /// <summary>
        /// Generates an NdArray with all elements set to the value 1
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with all elements set to the value 1</returns>
        NdArray<T> Ones(long size);
        /// <summary>
        /// Generates an NdArray with all elements set to the value 0
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with all elements set to the value 0</returns>
        NdArray<T> Zeroes(long size);
        /// <summary>
        /// Generates an NdArray with all elements set to the given value
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <param name="value">The value to initialize the elements with</param>
        /// <returns>An NdArray with all elements set to the given value</returns>
        NdArray<T> Same(T value, long size);
        /// <summary>
        /// Generates an NdArray with uninitialized data
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with uninitialized</returns>
        NdArray<T> Empty(long size);
        /// <summary>
        /// Generates an NdArray with all elements set to a random value
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with all elements set to a random value</returns>
        NdArray<T> Random(long size);

        /// <summary>
        /// Generates an NdArray with sequential integers, starting with zero
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>A shaped NdArray with sequential integers, starting with zero</returns>
        NdArray<T> Arange(Shape shape);
        /// <summary>
        /// Generates an NdArray with all elements set to the value 1
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>A shaped NdArray with all elements set to the value 1</returns>
        NdArray<T> Ones(Shape shape);
        /// <summary>
        /// Generates an NdArray with all elements set to the value 0
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>A shaped NdArray with all elements set to the value 0</returns>
        NdArray<T> Zeroes(Shape shape);
        /// <summary>
        /// Generates an NdArray with all elements set to the given value
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <param name="value">The value to initialize the elements with</param>
        /// <returns>A shaped NdArray with all elements set to the given value</returns>
        NdArray<T> Same(T value, Shape shape);
        /// <summary>
        /// Generates an NdArray with all elements set to a random value
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>A shaped NdArray with all elements set to a random value</returns>
        NdArray<T> Random(Shape shape);
    }

	/// <summary>
	/// Marker interface for random ops
	/// </summary>
	public interface IRandomGeneratorOp<T> : INullaryOp<T>
	{ }

	/// <summary>
	/// Random generator for sbyte.
	/// </summary>
	public struct RandomGeneratorOpSByte : IRandomGeneratorOp<sbyte>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public sbyte Op() { return (sbyte)(Rand.Next() & 0x7f); }
	}

	/// <summary>
	/// Random generator for byte.
	/// </summary>
	public struct RandomGeneratorOpByte : IRandomGeneratorOp<byte>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public byte Op() { return (byte)(Rand.Next() & 0xff); }
	}

	/// <summary>
	/// Random generator for short.
	/// </summary>
	public struct RandomGeneratorOpInt16 : IRandomGeneratorOp<short>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public short Op() { return (short)(Rand.Next() & 0x7fff); }
	}

	/// <summary>
	/// Random generator for ushort.
	/// </summary>
	public struct RandomGeneratorOpUInt16 : IRandomGeneratorOp<ushort>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public ushort Op() { return (ushort)(Rand.Next() & 0xffff); }
	}

	/// <summary>
	/// Random generator for int.
	/// </summary>
	public struct RandomGeneratorOpInt32 : IRandomGeneratorOp<int>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public int Op() { return (int)(Rand.Next() & 0x7fffffff); }
	}

	/// <summary>
	/// Random generator for uint.
	/// </summary>
	public struct RandomGeneratorOpUInt32 : IRandomGeneratorOp<uint>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public uint Op() 
		{
            uint upper = ((uint)Rand.Next() & 0xffff);
            uint lower = ((uint)Rand.Next() & 0xffff);

            return (uint)((upper << 16) | lower);
		}
	}

	/// <summary>
	/// Random generator for long.
	/// </summary>
	public struct RandomGeneratorOpInt64 : IRandomGeneratorOp<long>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public long Op() 
		{
	        ulong upper = (ulong)(Rand.Next() & 0x7ffffff);
	        ulong lower = (ulong)(Rand.Next());

	        return (long)((upper << 32) | lower);
		}
	}

	/// <summary>
	/// Random generator for ulong.
	/// </summary>
	public struct RandomGeneratorOpUInt64 : IRandomGeneratorOp<ulong>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public ulong Op() 
		{
	        uint a1 = ((uint)Rand.Next() & 0xffffff);
	        uint a2 = ((uint)Rand.Next() & 0xffffff);
	        uint a3 = ((uint)Rand.Next() & 0xffff);

	        return (ulong)((a3 << 40) | (a2 << 16) | a1);
		}
	}

	/// <summary>
	/// Random generator for float.
	/// </summary>
	public struct RandomGeneratorOpSingle : IRandomGeneratorOp<float>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public float Op() { return (float)Rand.NextDouble(); }
	}

	/// <summary>
	/// Random generator for double.
	/// </summary>
	public struct RandomGeneratorOpDouble : IRandomGeneratorOp<double>
	{
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

	    /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public double Op() { return (double)Rand.NextDouble(); }
	}

    /// <summary>
    /// Basic generator implementation that just calls "Set(x)" on the NdArray
    /// </summary>
    /// <typeparam name="T">The type of data to generate</typeparam>
	/// <typeparam name="C">The random number generator to use</typeparam>
    public class Generator<T, C> : IGenerator<T>
		where C : struct, IRandomGeneratorOp<T>
    {
        /// <summary>
        /// Generates an NdArray with sequential integers, starting with zero
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with sequential integers, starting with zero</returns>
        public NdArray<T> Arange(long size) { return RangeGenerator<T>.Generate(size); }
        /// <summary>
        /// Generates an NdArray with all elements set to the value 1
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with all elements set to the value 1</returns>
        public NdArray<T> Ones(long size) { return GenerateSame(size, (T)Convert.ChangeType(1, typeof(T))); }
        /// <summary>
        /// Generates an NdArray with all elements set to the value 0
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with all elements set to the value 0</returns>
        public NdArray<T> Zeroes(long size) { return GenerateSame(size, (T)Convert.ChangeType(0, typeof(T))); ; }
        /// <summary>
        /// Generates an NdArray with all elements set to the given value
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <param name="value">The value to initialize the elements with</param>
        /// <returns>An NdArray with all elements set to the given value</returns>
        public NdArray<T> Same(T value, long size) { return GenerateSame(size, value); }
        /// <summary>
        /// Generates an NdArray with uninitialized data
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with uninitialized</returns>
        public NdArray<T> Empty(long size) { return new NdArray<T>(new Shape(size)); }
        /// <summary>
        /// Generates an NdArray with all elements set to a random value
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with all elements set to a random value</returns>
        public virtual NdArray<T> Random(long size) 
        {
            var x = new NdArray<T>(new Shape(size));
            UFunc.Apply<T, C>(new C(), x);
            return x;
        }

        /// <summary>
        /// Generates an NdArray with sequential integers, starting with zero
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>A shaped NdArray with sequential integers, starting with zero</returns>
        public NdArray<T> Arange(Shape shape) { return RangeGenerator<T>.Generate(shape); }
        /// <summary>
        /// Generates an NdArray with all elements set to the value 1
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>A shaped NdArray with all elements set to the value 1</returns>
        public NdArray<T> Ones(Shape shape) { return GenerateSame(shape, (T)Convert.ChangeType(1, typeof(T))); }
        /// <summary>
        /// Generates an NdArray with all elements set to the value 0
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>A shaped NdArray with all elements set to the value 0</returns>
        public NdArray<T> Zeroes(Shape shape) { return GenerateSame(shape, default(T)); }
        /// <summary>
        /// Generates an NdArray with all elements set to the given value
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <param name="value">The value to initialize the elements with</param>
        /// <returns>A shaped NdArray with all elements set to the given value</returns>
        public NdArray<T> Same(T value, Shape shape) { return GenerateSame(shape, value); }
        /// <summary>
        /// Generates an NdArray with uninitialized data
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>An NdArray with uninitialized</returns>
        public NdArray<T> Empty(Shape shape) { return new NdArray<T>(shape); }
        /// <summary>
        /// Generates an NdArray with all elements set to a random value
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>An NdArray with all elements set to a random value</returns>
        public virtual NdArray<T> Random(Shape shape)
        {
            var x = new NdArray<T>(shape);
            UFunc.Apply<T, C>(new C(), x);
            return x;
        }

        /// <summary>
        /// Generates an NdArray with all elements set to the given value
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <param name="value">The value to initialize the elements with</param>
        /// <returns>A shaped NdArray with all elements set to the given value</returns>
        public static NdArray<T> GenerateSame(Shape shape, T value)
        {
            var x = new NdArray<T>(shape);
            x.Set(value);
            return x;
        }

        /// <summary>
        /// Generates an NdArray with all elements set to the given value
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <param name="value">The value to initialize the elements with</param>
        /// <returns>An NdArray with all elements set to the given value</returns>
        public static NdArray<T> GenerateSame(long size, T value)
        {
            var x = new NdArray<T>(new Shape(size));
            x.Set(value);
            return x;
        }
    }

    /// <summary>
    /// Interface for generating values
    /// </summary>
    /// <typeparam name="T">The type of data to generate</typeparam>
    public interface IGeneratorImplementation<T>
    {
        /// <summary>
        /// Generates an NdArray of the given size
        /// </summary>
        /// <param name="size">The size of the NdArray to generate</param>
        /// <returns>An NdArray of the given size</returns>
        NdArray<T> Generate(long size);
        /// <summary>
        /// Generates an NdArray with the given shape
        /// </summary>
        /// <param name="shape">The shape of the NdArray to generate</param>
        /// <returns>An NdArray with the given shape</returns>
        NdArray<T> Generate(Shape shape);
    }

    /// <summary>
    /// Basic implementation of the range generator
    /// </summary>
    /// <typeparam name="T">The type of data to generate</typeparam>
    public class RangeGenerator<T> : IGeneratorImplementation<T>
    {
        /// <summary>
        /// Generates an NdArray with sequential integers, starting with 0
        /// </summary>
        /// <param name="shape">The shape of the NdArray to generate</param>
        /// <returns>A shaped NdArray with sequential integers, starting with 0</returns>
        public static NdArray<T> Generate(Shape shape)
        {
            return Generate(shape.Length).Reshape(shape);
        }

        /// <summary>
        /// Generates an NdArray with sequential integers, starting with 0
        /// </summary>
        /// <param name="size">The size of the NdArray to generate</param>
        /// <returns>An NdArray with sequential integers, starting with 0</returns>
        public static NdArray<T> Generate(long size)
        {
            T[] a = new T[size];
            long value = 0;

            if (size <= int.MaxValue)
            {
                for (int i = 0; i < a.Length; i++)
                    a[i] = (T)Convert.ChangeType(value++, typeof(T));
            }
            else
            {
                for (long i = 0; i < a.LongLength; i++)
                    a[i] = (T)Convert.ChangeType(value++, typeof(T));
            }

            return new NdArray<T>(a);
        }

        #region IGenerator<T> Members

        /// <summary>
        /// Generates an NdArray with sequential integers, starting with 0
        /// </summary>
        /// <param name="size">The size of the NdArray to generate</param>
        /// <returns>An NdArray with sequential integers, starting with 0</returns>
        NdArray<T> NumCIL.Generic.IGeneratorImplementation<T>.Generate(long size)
        {
            return RangeGenerator<T>.Generate(size);
        }

        /// <summary>
        /// Generates an NdArray with sequential integers, starting with 0
        /// </summary>
        /// <param name="shape">The shape of the NdArray to generate</param>
        /// <returns>A shaped NdArray with sequential integers, starting with 0</returns>
        NdArray<T> NumCIL.Generic.IGeneratorImplementation<T>.Generate(Shape shape)
        {
            return RangeGenerator<T>.Generate(shape);
        }

        #endregion
    }
}
