#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium:
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
        NdArray<T> Range(long size);
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
        /// Generates an NdArray with uninitialized data
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>An NdArray with uninitialized</returns>
        NdArray<T> Empty(Shape shape);
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
        NdArray<T> Range(Shape shape);
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
    public interface IRandomGeneratorOp<T> : INullaryOp<T>, IRandomOp
	{ }

    /// <summary>
    /// Non-generic marker interface for random operators
    /// </summary>
    public interface IRandomOp { }

	/// <summary>
	/// Random generator for sbyte.
	/// </summary>
	public struct RandomGeneratorOpInt8 : IRandomGeneratorOp<sbyte>
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
	public struct RandomGeneratorOpUInt8 : IRandomGeneratorOp<byte>
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
	public struct RandomGeneratorOpFloat : IRandomGeneratorOp<float>
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
    /// Random generator for bool.
    /// </summary>
    public struct RandomGeneratorOpBoolean : IRandomGeneratorOp<bool>
    {
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

        /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public bool Op() { return (Rand.Next() & 0x1) == 1; }
    }

    /// <summary>
    /// Random generator for Complex64.
    /// </summary>
    public struct RandomGeneratorOpComplex64 : IRandomGeneratorOp<NumCIL.Complex64.DataType>
    {
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

        /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public NumCIL.Complex64.DataType Op() { return (NumCIL.Complex64.DataType)(float)Rand.NextDouble(); }
    }

    /// <summary>
    /// Random generator for Complex128.
    /// </summary>
    public struct RandomGeneratorOpComplex128 : IRandomGeneratorOp<System.Numerics.Complex>
    {
        /// <summary>Private reference to an initialized random number generator</summary>
        private static readonly System.Random Rand = new System.Random();

        /// <summary>Returns a random number</summary>
        /// <returns>A random number</returns>
        public System.Numerics.Complex Op() { return (System.Numerics.Complex)Rand.NextDouble(); }
    }

    /// <summary>
    /// Basic generator implementation that just calls "Set(x)" on the NdArray
    /// </summary>
    /// <typeparam name="T">The type of data to generate</typeparam>
	/// <typeparam name="TRand">The random number generator to use</typeparam>
    /// <typeparam name="TConv">The conversion operator to use</typeparam>
    public class Generator<T, TRand, TRange, TConv> : IGenerator<T>
        where TRand : struct, IRandomGeneratorOp<T>
        where TRange : struct, IRangeGeneratorOp<T>
        where TConv : struct, INumberConverter<T>
    {
        /// <summary>
        /// Generates an NdArray with sequential integers, starting with zero
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with sequential integers, starting with zero</returns>
        public NdArray<T> Range(long size) 
        { 
            var x = new NdArray<T>(new Shape(size));
            UFunc.Apply<T, TRange>(new TRange(), x);
            return x;
        }
        /// <summary>
        /// Generates an NdArray with all elements set to the value 1
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with all elements set to the value 1</returns>
        public NdArray<T> Ones(long size) { return GenerateSame(size, new TConv().Convert(1)); }
        /// <summary>
        /// Generates an NdArray with all elements set to the value 0
        /// </summary>
        /// <param name="size">The length of the generated array</param>
        /// <returns>An NdArray with all elements set to the value 0</returns>
        public NdArray<T> Zeroes(long size) { return GenerateSame(size, new TConv().Convert(0)); }
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
            UFunc.Apply<T, TRand>(new TRand(), x);
            return x;
        }

        /// <summary>
        /// Generates an NdArray with sequential integers, starting with zero
        /// </summary>
        /// <param name="shape">The shape of the generated array</param>
        /// <returns>A shaped NdArray with sequential integers, starting with zero</returns>
        public NdArray<T> Range(Shape shape) 
        { 
            var x = new NdArray<T>(shape);
            UFunc.Apply<T, TRange>(new TRange(), x);
            return x;
        }
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
            UFunc.Apply<T, TRand>(new TRand(), x);
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
    /// Interfaces that describes a single function for converting a long to another type
    /// </summary>
    /// <typeparam name="T">The type to convert to</typeparam>
    public interface INumberConverter<T>
    {
        /// <summary>
        /// Converts the value to a type
        /// </summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        T Convert(long value);
    }

    /// <summary>
    /// Implementation of a number conversion to sbyte
    /// </summary>
    public struct NumberConverterInt8 : INumberConverter<sbyte>
    {
        /// <summary>Converts the value to a <see cref="System.SByte"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public sbyte Convert(long value) { return (sbyte)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to byte
    /// </summary>
    public struct NumberConverterUInt8 : INumberConverter<byte>
    {
        /// <summary>Converts the value to a <see cref="System.Byte"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public byte Convert(long value) { return (byte)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to short
    /// </summary>
    public struct NumberConverterInt16 : INumberConverter<short>
    {
        /// <summary>Converts the value to a <see cref="System.Int16"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public short Convert(long value) { return (short)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to ushort
    /// </summary>
    public struct NumberConverterUInt16 : INumberConverter<ushort>
    {
        /// <summary>Converts the value to a <see cref="System.UInt16"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public ushort Convert(long value) { return (ushort)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to int
    /// </summary>
    public struct NumberConverterInt32 : INumberConverter<int>
    {
        /// <summary>Converts the value to a <see cref="System.Int32"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public int Convert(long value) { return (int)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to uint
    /// </summary>
    public struct NumberConverterUInt32 : INumberConverter<uint>
    {
        /// <summary>Converts the value to a <see cref="System.UInt32"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public uint Convert(long value) { return (uint)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to long
    /// </summary>
    public struct NumberConverterInt64 : INumberConverter<long>
    {
        /// <summary>Converts the value to a <see cref="System.Int64"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public long Convert(long value) { return value; }
    }

    /// <summary>
    /// Implementation of a number conversion to ulong
    /// </summary>
    public struct NumberConverterUInt64 : INumberConverter<ulong>
    {
        /// <summary>Converts the value to a <see cref="System.UInt64"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public ulong Convert(long value) { return (ulong)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to float
    /// </summary>
    public struct NumberConverterFloat : INumberConverter<float>
    {
        /// <summary>Converts the value to a <see cref="System.Single"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public float Convert(long value) { return (float)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to double
    /// </summary>
    public struct NumberConverterDouble : INumberConverter<double>
    {
        /// <summary>Converts the value to a <see cref="System.Double"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public double Convert(long value) { return (double)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to bool
    /// </summary>
    public struct NumberConverterBoolean : INumberConverter<bool>
    {
        /// <summary>Converts the value to a <see cref="System.Boolean"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public bool Convert(long value) { return !(value == 0); }
    }

    /// <summary>
    /// Implementation of a number conversion to complex64
    /// </summary>
    public struct NumberConverterComplex64 : INumberConverter<NumCIL.Complex64.DataType>
    {
        /// <summary>Converts the value to a <see cref="NumCIL.Complex64.DataType"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public NumCIL.Complex64.DataType Convert(long value) { return (NumCIL.Complex64.DataType)value; }
    }

    /// <summary>
    /// Implementation of a number conversion to complex128
    /// </summary>
    public struct NumberConverterComplex128 : INumberConverter<System.Numerics.Complex>
    {
        /// <summary>Converts the value to a <see cref="System.Numerics.Complex"/></summary>
        /// <param name="value">The number to convert</param>
        /// <returns>The converted value</returns>
        public System.Numerics.Complex Convert(long value) { return (System.Numerics.Complex)value; }
    }

    /// <summary>
    /// Marker interface for range ops
    /// </summary>
    public interface IRangeGeneratorOp<T> : INullaryOp<T>, IRangeOp
    { }

    /// <summary>
    /// Non-generic marker interface for range operators
    /// </summary>
    public interface IRangeOp { }

    /// <summary>
    /// Range generator.
    /// </summary>
    public struct RangeGeneratorOp<T, TConv> : IRangeGeneratorOp<T>
        where TConv : struct, INumberConverter<T>
    {
        /// <summary>
        /// The internal counter
        /// </summary>
        private long m_no;

        /// <summary>
        /// Cached converter &quot;instance&quot;
        /// </summary>
        private TConv m_conv;

        /// <summary>Returns a range number</summary>
        /// <returns>A range number</returns>
        public T Op() { return m_conv.Convert(m_no++); }
    }
}
