using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.UInt32
{
    using T = System.UInt32;
    using InArray = NumCIL.Generic.NdArray<System.UInt32>;
    using OutArray = NdArray;

    /// <summary>
    /// A wrapper for a basic NdArray to a similar one with typed operators
    /// </summary>
    public partial class NdArray : IEnumerable<InArray>
    {
        #region NdArray Mimics
        /// <summary>
        /// The value instance that gives access to values
        /// </summary>
        public InArray.ValueAccessor Value { get { return value.Value; } }
        /// <summary>
        /// A reference to the shape instance that describes this view
        /// </summary>
        public Shape Shape { get { return value.Shape; } }
        /// <summary>
        /// A reference to the underlying data storage
        /// </summary>
        public T[] Data { get { return value.Data; } }
        /// <summary>
        /// Gets a subview on the array
        /// </summary>
        /// <param name="index">The element to get the view from</param>
        /// <returns>A view on the selected element</returns>
        public OutArray this[params long[] index] { get { return this.value[index]; } set { this.value[index] = value; } }
        /// <summary>
        /// Gets a subview on the array
        /// </summary>
        /// <param name="ranges">The range get the view from</param>
        /// <returns>A view on the selected element</returns>
        public OutArray this[params Range[] ranges] { get { return this.value[ranges]; } set { this.value[ranges] = value; } }
        /// <summary>
        /// Returns a flattened (1-d copy) of the current data view
        /// </summary>
        /// <returns>A flattened copy</returns>
        public OutArray Flatten() { return this.value.Flatten(); }
        /// <summary>
        /// Generates a new view based on this array
        /// </summary>
        /// <param name="newshape">The new shape</param>
        /// <returns>The reshaped array</returns>
        public OutArray Reshape(Shape newshape) { return this.value.Reshape(newshape); }
        /// <summary>
        /// Returns a view that is a view of a single element
        /// </summary>
        /// <param name="element">The element to view</param>
        /// <returns>The subview</returns>
        public OutArray Subview(long element) { return this.value.Subview(element); }
        /// <summary>
        /// Returns a view that is a view of a range of elements
        /// </summary>
        /// <param name="element">The range to view</param>
        /// <returns>The subview</returns>
        public OutArray Subview(Range range, long dimension) { return this.value.Subview(range, dimension); }
        /// <summary>
        /// Returns an enumerator that iterates through a collection.
        /// </summary>
        /// <returns>
        /// An <see cref="T:System.Collections.Generic.IEnumerator"/> object that can be used to iterate through the collection.
        /// </returns>
        public IEnumerator<InArray> GetEnumerator() { return this.value.GetEnumerator(); }
        /// <summary>
        /// Returns an enumerator that iterates through a collection.
        /// </summary>
        /// <returns>
        /// An <see cref="T:System.Collections.IEnumerator"/> object that can be used to iterate through the collection.
        /// </returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() { return this.value.GetEnumerator(); }
        /// <summary>
        /// Constructs a NdArray that is a scalar wrapper,
        /// allows simple scalar operations on arbitrary
        /// NdArrays
        /// </summary>
        /// <param name="value">The scalar value</param>
        public NdArray(T value)
            : this(new T[] { value }, new long[] { 1 })
        {
        }

        /// <summary>
        /// Constructs a new NdArray over a pre-allocated array
        /// </summary>
        /// <param name="data">The data to wrap in a NdArray</param>
        public NdArray(Shape shape)
            : this(new T[shape.Length], shape)
        {
        }

        /// <summary>
        /// Constructs a new NdArray over a pre-allocated array and shapes it
        /// </summary>
        /// <param name="data">The data to wrap in a NdArray</param>
        /// <param name="shape">The shape to view the array in</param>
        public NdArray(T[] data, Shape shape = null)
            : this(new InArray(data, shape))
        {
        }

        /// <summary>
        /// Constructs a new NdArray over a pre-allocated array and shapes it
        /// </summary>
        /// <param name="source">An existing array that will be re-shaped</param>
        /// <param name="shape">The shape to view the array in</param>
        public NdArray(InArray source, Shape newshape)
            : this(new InArray(source, newshape))
        {
        }

        /// <summary>
        /// Returns a <see cref="System.String"/> that represents this instance.
        /// </summary>
        /// <returns>
        /// A <see cref="System.String"/> that represents this instance.
        /// </returns>
        public override string ToString()
        {
            return this.value.ToString();
        }
        #endregion

        /// <summary>
        /// The only data member of the struct is a reference to the underlying view
        /// </summary>
        private readonly InArray value;
        /// <summary>
        /// Constructs a new typed array from a basic one
        /// </summary>
        /// <param name="v">The basic array</param>
        public NdArray(InArray v) { this.value = v; }

        #region Implict conversion operators
        /// <summary>
        /// Implicit operator that returns a typed array from a basic one
        /// </summary>
        /// <param name="v">The basic array to wrap</param>
        /// <returns>A wrapped array</returns>
        public static implicit operator OutArray(InArray v) { return new OutArray(v); }
        /// <summary>
        /// Implicit operator that returns a basic array from a wrapped one
        /// </summary>
        /// <param name="v">The wrapper array</param>
        /// <returns>The basic array</returns>
        public static implicit operator InArray(NdArray v) { return v.value; }
        /// <summary>
        /// Implicit operator that returns a wrapped array from a scalar value
        /// </summary>
        /// <param name="v">The scalar value</param>
        /// <returns>A wrapped array representing the scalar value</returns>
        public static implicit operator OutArray(T v) { return new OutArray(v); }
        #endregion

        #region Operator implementations
        public static OutArray operator +(OutArray a, OutArray b) { return UFunc.Apply<T, Add>(a, b, null); }
        public static OutArray operator -(OutArray a, OutArray b) { return UFunc.Apply<T, Sub>(a, b, null); }
        public static OutArray operator *(OutArray a, OutArray b) { return UFunc.Apply<T, Mul>(a, b, null); }
        public static OutArray operator /(OutArray a, OutArray b) { return UFunc.Apply<T, Div>(a, b, null); }
        public static OutArray operator %(OutArray a, OutArray b) { return UFunc.Apply<T, Mod>(a, b, null); }

        public static OutArray operator +(OutArray a, T b) { return UFunc.Apply<T, Add>(a, b, null); }
        public static OutArray operator -(OutArray a, T b) { return UFunc.Apply<T, Sub>(a, b, null); }
        public static OutArray operator *(OutArray a, T b) { return UFunc.Apply<T, Mul>(a, b, null); }
        public static OutArray operator /(OutArray a, T b) { return UFunc.Apply<T, Div>(a, b, null); }
        public static OutArray operator %(OutArray a, T b) { return UFunc.Apply<T, Mod>(a, b, null); }

        public static OutArray operator ++(OutArray a) { return UFunc.Apply<T, Inc>(a); }
        public static OutArray operator --(OutArray a) { return UFunc.Apply<T, Inc>(a); }
        #endregion

        #region Common function implementations
        /// <summary>
        /// Returns a new array with the maximum value for each element
        /// </summary>
        /// <param name="a">One array</param>
        /// <param name="b">Another array</param>
        /// <returns>An array with the maximum values for both arrays</returns>
        public static OutArray Max(OutArray a, OutArray b) { return UFunc.Apply<T, Max>(a, b, null); }
        /// <summary>
        /// Returns a new array with the minimum value for each element
        /// </summary>
        /// <param name="a">One array</param>
        /// <param name="b">Another array</param>
        /// <returns>An array with the minimum values for both arrays</returns>
        public static OutArray Min(OutArray a, OutArray b) { return UFunc.Apply<T, Min>(a, b, null); }

        public static OutArray Ceiling(OutArray a) { return UFunc.Apply<T, Ceiling>(a); }
        public static OutArray Floor(OutArray a) { return UFunc.Apply<T, Floor>(a); }
        public static OutArray Round(OutArray a) { return UFunc.Apply<T, Round>(a); }
        public static OutArray Abs(OutArray a) { return UFunc.Apply<T, Abs>(a); }
        public static OutArray Sqrt(OutArray a) { return UFunc.Apply<T, Sqrt>(a); }

        public OutArray Ceiling() { return UFunc.Apply<T, Ceiling>(this); }
        public OutArray Floor() { return UFunc.Apply<T, Floor>(this); }
        public OutArray Round() { return UFunc.Apply<T, Round>(this); }
        public OutArray Abs() { return UFunc.Apply<T, Abs>(this); }
        public OutArray Sqrt() { return UFunc.Apply<T, Sqrt>(this); }
        public OutArray Exp() { return UFunc.Apply<T, Exp>(this); }
        public OutArray Negate() { return UFunc.Apply<T, Negate>(this); }
        public OutArray Log() { return UFunc.Apply<T, Log>(this); }
        public OutArray Log10() { return UFunc.Apply<T, Log10>(this); }
        public OutArray Pow(T value) { return UFunc.Apply<T, Pow>(this, value, null); }
        #endregion


        #region Appliers for custom UFuncs
        public OutArray Apply(Func<T, T> op) { return UFunc.Apply(op, this); }
        public OutArray Apply(Func<T, T, T> op, InArray b) { return UFunc.Apply<T>(op, this, b); }
        public OutArray Apply<C>() where C : struct, IUnaryOp<T> { return UFunc.Apply<T, C>(this); }
        public OutArray Apply<C>(InArray b) where C : struct, IBinaryOp<T> { return UFunc.Apply<T, C>(this, b, null); }

        public OutArray Reduce<C>(long axis = 0) where C : struct, IBinaryOp<T> { return UFunc.Reduce<T, C>(this, axis, null); }
        #endregion

    }

    #region Operator implementations
    public struct Add : IBinaryOp<T> { public T Op(T a, T b) { return (T)(a + b); } }
    public struct Sub : IBinaryOp<T> { public T Op(T a, T b) { return (T)(a - b); } }
    public struct Mul : IBinaryOp<T> { public T Op(T a, T b) { return (T)(a * b); } }
    public struct Div : IBinaryOp<T> { public T Op(T a, T b) { return (T)(a / b); } }
    public struct Mod : IBinaryOp<T> { public T Op(T a, T b) { return (T)(a % b); } }

    public struct Max : IBinaryOp<T> { public T Op(T a, T b) { return (T)Math.Max(a, b); } }
    public struct Min : IBinaryOp<T> { public T Op(T a, T b) { return (T)Math.Min(a, b); } }

    public struct Inc : IUnaryOp<T> { public T Op(T a) { return (T)(a + (T)1); } }
    public struct Dec : IUnaryOp<T> { public T Op(T a) { return (T)(a + (T)1); } }

    public struct Ceiling : IUnaryOp<T> { public T Op(T a) { return (T)Math.Ceiling((double)a); } }
    public struct Floor : IUnaryOp<T> { public T Op(T a) { return (T)Math.Floor((double)a); } }
    public struct Round : IUnaryOp<T> { public T Op(T a) { return (T)Math.Round((double)a); } }
    public struct Abs : IUnaryOp<T> { public T Op(T a) { return (T)Math.Abs(a); } }
    public struct Sqrt : IUnaryOp<T> { public T Op(T a) { return (T)Math.Sqrt(a); } }
    public struct Exp : IUnaryOp<T> { public T Op(T a) { return (T)Math.Exp(a); } }
    public struct Negate : IUnaryOp<T> { public T Op(T a) { return (T)(-a); } }
    public struct Log : IUnaryOp<T> { public T Op(T a) { return (T)Math.Log(a); } }
    public struct Log10 : IUnaryOp<T> { public T Op(T a) { return (T)Math.Log10(a); } }
    public struct Pow : IBinaryOp<T> { public T Op(T a, T b) { return (T)Math.Pow(a, b); } }
    public struct Random : IUnaryOp<T> { public T Op(T a) { return (T)NumCIL.Generic.Generator<T>.Rand.NextDouble(); } }
    #endregion

    #region Generate mimics
    public static class Generate
    {
        public static NumCIL.Generic.Generator<T> Generator = new NumCIL.Generic.Generator<T>();

        public static OutArray Arange(Shape shape) { return Generator.Arange(shape); }
        public static OutArray Ones(Shape shape) { return Generator.Ones(shape); }
        public static OutArray Zeroes(Shape shape) { return Generator.Zeroes(shape); }
        public static OutArray Empty(Shape shape) { return Generator.Empty(shape); }
        public static OutArray Same(T value, Shape shape) { return Generator.Same(value, shape); }
        public static OutArray Random(Shape shape) { var op = Generator.Empty(shape); UFunc.Apply<T, Random>(op, op); return op; }

        public static OutArray Arange(params long[] dimensions) { return Generator.Arange(dimensions); }
        public static OutArray Ones(params long[] dimensions) { return Generator.Ones(dimensions); }
        public static OutArray Zeroes(params long[] dimensions) { return Generator.Zeroes(dimensions); }
        public static OutArray Empty(params long[] dimensions) { return Generator.Empty(dimensions); }
        public static OutArray Same(T value, params long[] dimensions) { return Generator.Same(value, dimensions); }
        public static OutArray Random(params long[] dimensions) { var op = Generator.Empty(dimensions); UFunc.Apply<T, Random>(op, op); return op; }
    }
    #endregion
}


