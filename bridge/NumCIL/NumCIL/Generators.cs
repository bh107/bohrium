using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.Generic
{

    public interface IGenerator<T>
    {
        NdArray<T> Arange(long size);
        NdArray<T> Ones(long size);
        NdArray<T> Zeroes(long size);
        NdArray<T> Same(T value, long size);
        NdArray<T> Empty(long size);

        NdArray<T> Arange(Shape shape);
        NdArray<T> Ones(Shape shape);
        NdArray<T> Zeroes(Shape shape);
        NdArray<T> Same(T value, Shape shape);
        NdArray<T> Empty(Shape shape);
    }

    public class Generator<T> : IGenerator<T>
    {
        public static System.Random Rand = new System.Random();

        public NdArray<T> Arange(long size) { return RangeGenerator<T>.Generate(size); }
        public NdArray<T> Ones(long size) { return GenerateSame(size, (T)Convert.ChangeType(1, typeof(T))); }
        public NdArray<T> Zeroes(long size) { return GenerateSame(size, (T)Convert.ChangeType(0, typeof(T))); ; }
        public NdArray<T> Same(T value, long size) { return GenerateSame(size, value); }
        public NdArray<T> Empty(long size) { return new NdArray<T>(new Shape(size)); }

        public NdArray<T> Arange(Shape shape) { return RangeGenerator<T>.Generate(shape); }
        public NdArray<T> Ones(Shape shape) { return GenerateSame(shape, (T)Convert.ChangeType(1, typeof(T))); }
        public NdArray<T> Zeroes(Shape shape) { return GenerateSame(shape, default(T)); }
        public NdArray<T> Same(T value, Shape shape) { return GenerateSame(shape, value); }
        public NdArray<T> Empty(Shape shape) { return new NdArray<T>(shape); }

        public static NdArray<T> GenerateSame(Shape shape, T value)
        {
            var x = new NdArray<T>(shape);
            x.Set(value);
            return x;
        }

        public static NdArray<T> GenerateSame(long size, T value)
        {
            var x = new NdArray<T>(new Shape(size));
            x.Set(value);
            return x;
        }
    }

    public interface IGeneratorImplementation<T>
    {
        NdArray<T> Generate(long size);
        NdArray<T> Generate(Shape shape);
    }

    public class RangeGenerator<T> : IGeneratorImplementation<T>
    {
        public static NdArray<T> Generate(Shape shape)
        {
            return Generate(shape.Length).Reshape(shape);
        }
        
        public static NdArray<T> Generate(long size)
        {
            T[] a = new T[size];
            long value = 0;

            if (size <= int.MaxValue)
            {
                for (long i = 0; i < a.Length; i++)
                    a[i] = (T)Convert.ChangeType(value++, typeof(T));
            }
            else
            {
                for (long i = 0; i < size; i++)
                    a[i] = (T)Convert.ChangeType(value++, typeof(T));
            }

            return new NdArray<T>(a);
        }

        #region IGenerator<T> Members

        NdArray<T> NumCIL.Generic.IGeneratorImplementation<T>.Generate(long size)
        {
            return RangeGenerator<T>.Generate(size);
        }

        NdArray<T> NumCIL.Generic.IGeneratorImplementation<T>.Generate(Shape shape)
        {
            return RangeGenerator<T>.Generate(shape);
        }

        #endregion
    }
}
