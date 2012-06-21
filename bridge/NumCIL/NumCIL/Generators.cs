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
    /// Implementation of a basic random number generator
    /// </summary>
    /// <typeparam name="T">The type of data to generate</typeparam>
    public struct RandomGeneratorOp<T> : INullaryOp<T>
    {
        /// <summary>
        /// Private reference to an initialized random number generator
        /// </summary>
        private static readonly System.Random Rand = new System.Random();

        /// <summary>
        /// Returns a random number appropriate for the basic numeric types
        /// </summary>
        /// <returns>A random number</returns>
        public T Op()
        {
            if (typeof(T) == typeof(sbyte))
                return (T)(object)(sbyte)(Rand.Next() & 0x7f);
            else if (typeof(T) == typeof(short))
                return (T)(object)(short)(Rand.Next() & 0x7fff);
            else if (typeof(T) == typeof(int))
                return (T)(object)(int)(Rand.Next() & 0x7fffffff);
            else if (typeof(T) == typeof(long))
            {
                ulong upper = (ulong)(Rand.Next() & 0x7ffffff);
                ulong lower = (ulong)(Rand.Next());

                return (T)(object)(long)((upper << 32) | lower);
            }
            else if (typeof(T) == typeof(byte))
                return (T)(object)(byte)(Rand.Next() & 0xff);
            else if (typeof(T) == typeof(ushort))
                return (T)(object)(ushort)(Rand.Next() & 0xffff);
            else if (typeof(T) == typeof(uint))
            {
                uint upper = ((uint)Rand.Next() & 0xffff);
                uint lower = ((uint)Rand.Next() & 0xffff);

                return (T)(object)(uint)((upper << 16) | lower);
            }
            else if (typeof(T) == typeof(ulong))
            {
                uint a1 = ((uint)Rand.Next() & 0xffffff);
                uint a2 = ((uint)Rand.Next() & 0xffffff);
                uint a3 = ((uint)Rand.Next() & 0xffff);

                return (T)(object)(ulong)((a3 << 40) | (a2 << 16) | a1);
            }
            else if (typeof(T) == typeof(float))
                return (T)(object)(float)Rand.NextDouble();
            else if (typeof(T) == typeof(double))
                return (T)(object)(double)Rand.NextDouble();
            else
                throw new Exception(string.Format("Unable to generate random numbers for {0}", typeof(T).FullName));
        }
    }

    /// <summary>
    /// Basic generator implementation that just calls "Set(x)" on the NdArray
    /// </summary>
    /// <typeparam name="T">The type of data to generate</typeparam>
    public class Generator<T> : IGenerator<T>
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
        public NdArray<T> Random(long size) 
        {
            var x = new NdArray<T>(new Shape(size));
            UFunc.Apply<T, RandomGeneratorOp<T>>(new RandomGeneratorOp<T>(), x);
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
        public NdArray<T> Random(Shape shape)
        {
            var x = new NdArray<T>(shape);
            UFunc.Apply<T, RandomGeneratorOp<T>>(new RandomGeneratorOp<T>(), x);
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
