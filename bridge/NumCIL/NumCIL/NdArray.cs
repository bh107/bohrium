using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.Generic
{
    /// <summary>
    /// Class that represents a multidimensional array
    /// </summary>
    /// <typeparam name="T">The array element type</typeparam>
    public class NdArray<T> : IEnumerable<NdArray<T>>
    {
        /// <summary>
        /// The factory used to create accessors, defaults to creating DefaultAccessor instances
        /// </summary>
        public static IAccessorFactory<T> AccessorFactory = new DefaultAccessorFactory<T>();

        /// <summary>
        /// A wrapper that allows array style indexing of values
        /// </summary>
        public class ValueAccessor : IEnumerable<T>
        {
            /// <summary>
            /// The reference to the data
            /// </summary>
            private T[] m_data = null;

            /// <summary>
            /// The NdArray that owns this instance
            /// </summary>
            private readonly NdArray<T> m_parent;

            /// <summary>
            /// Constructs a new ValueAccessor
            /// </summary>
            /// <param name="parent">The parent to lock the wrapper to</param>
            internal ValueAccessor(NdArray<T> parent)
            {
                m_parent = parent;
            }

            /// <summary>
            /// Gets or sets a value in the array
            /// </summary>
            /// <param name="index">The indices to look up</param>
            /// <returns>The value read from the underlying array</returns>
            public T this[params long[] index]
            {
                get 
                {
                    if (m_data == null)
                        m_data = m_parent.Data;

                    return m_data[m_parent.Shape[index]]; 
                }
                set 
                {
                    if (m_data == null)
                        m_data = m_parent.Data;
                    m_data[m_parent.Shape[index]] = value; 
                }
            }

            #region IEnumerable<T> Members

            /// <summary>
            /// Returns an enumerator that iterates through a collection.
            /// </summary>
            /// <returns>
            /// An <see cref="T:System.Collections.Generic.IEnumerator"/> object that can be used to iterate through the collection.
            /// </returns>
            public IEnumerator<T> GetEnumerator()
            {
                return new Utility.AnyEnumerator<T>(index => this[index], m_parent.Shape.Dimensions[0].Length);
            }

            #endregion

            #region IEnumerable Members

            /// <summary>
            /// Returns an enumerator that iterates through a collection.
            /// </summary>
            /// <returns>
            /// An <see cref="T:System.Collections.IEnumerator"/> object that can be used to iterate through the collection.
            /// </returns>
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return this.GetEnumerator();
            }

            #endregion
        }

        /// <summary>
        /// A reference to the underlying data storage, should not be accessed directly, may not be completely updated
        /// </summary>
        public readonly IDataAccessor<T> m_data;

        /// <summary>
        /// Gets the real underlying data, accessing this property may flush pending executions
        /// </summary>
        public T[] Data
        {
            get 
            {
                return m_data.Data;
            }
        }


        /// <summary>
        /// A reference to the shape instance that describes this view
        /// </summary>
        public readonly Shape Shape;

        /// <summary>
        /// The value instance that gives access to values
        /// </summary>
        public readonly ValueAccessor Value;

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
        /// <param name="shape">The shape of the NdArray</param>
        public NdArray(Shape shape)
            : this(AccessorFactory.Create(shape.Length), shape)
        {
        }

        /// <summary>
        /// Constructs a new NdArray over a pre-allocated array and optionally shapes it
        /// </summary>
        /// <param name="data">The data to wrap in a NdArray</param>
        /// <param name="shape">The shape to view the array in</param>
        public NdArray(T[] data, Shape shape = null)
            : this(AccessorFactory.Create(data), shape)
        {
        }

        /// <summary>
        /// Constructs a new NdArray over an existing data block and optionally shapes it
        /// </summary>
        /// <param name="data">The data to wrap in a NdArray</param>
        /// <param name="shape">The shape to view the array in</param>
        public NdArray(IDataAccessor<T> data, Shape shape = null)
        {
            this.Shape = shape ?? new long[] { data.Length };
            if (data.Length < this.Shape.Length)
                throw new ArgumentOutOfRangeException("dimensionsizes");

            m_data = data;

            Value = new ValueAccessor(this);
        }

        /// <summary>
        /// Constructs a new NdArray over a pre-allocated array and shapes it
        /// </summary>
        /// <param name="source">An existing array that will be re-shaped</param>
        /// <param name="newshape">The shape to view the array in</param>
        public NdArray(NdArray<T> source, Shape newshape = null)
            : this(source.m_data, newshape)
        {
        }

        /// <summary>
        /// Generates a new view based on this array
        /// </summary>
        /// <param name="newshape">The new shape</param>
        /// <returns>The reshaped array</returns>
        public NdArray<T> Reshape(Shape newshape)
        {
            return new NdArray<T>(this, newshape);
        }

        /// <summary>
        /// Returns a view that is a view of a single element
        /// </summary>
        /// <param name="element">The element to view</param>
        /// <returns>The subview</returns>
        public NdArray<T> Subview(long element)
        {
            if (element < 0)
                element = this.Shape.Dimensions[0].Length + element;

            if (element < 0 || element > this.Shape.Dimensions[0].Length)
                throw new ArgumentOutOfRangeException("element");

            //Special case
            if (this.Shape.Dimensions.LongLength == 1)
            {
                long pos = this.Shape[element];
                return new NdArray<T>(this, new Shape(
                    new long[] { 1 }, //Single element
                    pos, //Offset to position
                    new long[] { this.Shape.Length - pos } //Skip the rest
                ));
            }

            Shape.ShapeDimension[] dims = new Shape.ShapeDimension[this.Shape.Dimensions.Length - 1];
            Array.Copy(this.Shape.Dimensions, 1, dims, 0, dims.Length);

            //We need to modify the top dimension to skip the elements we wish to hide
            long newoffset = this.Shape[element];
            return new NdArray<T>(this, new Shape(dims, newoffset));
        }

        /// <summary>
        /// Returns a view that is a view of a range of elements
        /// </summary>
        /// <param name="range">The range to view</param>
        /// <param name="dim">The dimension to apply the range to</param>
        /// <returns>The subview</returns>
        public NdArray<T> Subview(Range range, long dim)
        {
            long first = range.First < 0 ? (this.Shape.Dimensions[dim].Length) + range.First : range.First;
            long offset = dim == this.Shape.Dimensions.LongLength ? this.Shape.Length : this.Shape.Offset + first * this.Shape.Dimensions[dim].Stride;
            
            long last;
            long stride;

            if (range.Initialized)
            {
                if (range.NewDimension)
                {
                    Shape.ShapeDimension[] n = new Shape.ShapeDimension[this.Shape.Dimensions.LongLength + 1];
                    Array.Copy(this.Shape.Dimensions, 0, n, 0, dim);
                    Array.Copy(this.Shape.Dimensions, dim, n, dim + 1, n.LongLength - (dim + 1));
                    n[dim] = new NumCIL.Shape.ShapeDimension(1, 0);

                    return new NdArray<T>(this, new Shape(n, this.Shape.Offset));
                }
                else if (range.SingleElement)
                    last = first;
                else
                    last = range.Last <= 0 ? (this.Shape.Dimensions[dim].Length - 1) + range.Last : range.Last;

                stride = range.Stride;
            }
            else
            {
                last = this.Shape.Dimensions[dim].Length - 1;
                stride = 1;
            }


            long j = 0;
            
            var dimensions = this.Shape.Dimensions.Select(x =>
                {
                    if (j++ == dim)
                    {
                        if (range.Last == 0 && stride != 1)
                        {
                            long maxlast = last / stride;

                            return new Shape.ShapeDimension((maxlast - first) + 1, stride * x.Stride);
                        }
                        else
                            return new Shape.ShapeDimension((last - first) + 1, stride * x.Stride);
                    }
                    else
                        return x;
                }).ToArray();

            return new NdArray<T>(this, new Shape(dimensions, offset));
        }

        /// <summary>
        /// Gets a subview on the array
        /// </summary>
        /// <param name="index">The element to get the view from</param>
        /// <returns>A view on the selected element</returns>
        public NdArray<T> this[params long[] index]
        {
            get 
            {
                NdArray<T> v = this;
                foreach (long n in index)
                    v = v.Subview(n);
                
                return v;
            }
            set
            {
                NdArray<T> lv = this[index];

                //Self-assignment
                if (lv.Shape.Equals(value.Shape) && value.m_data == this.m_data)
                    return;


                if (lv.Shape.Dimensions.Length != value.Shape.Dimensions.Length)
                    throw new Exception("Cannot assign incompatible arrays");
                
                for(long i = 0; i < lv.Shape.Dimensions.Length; i++)
                    if (lv.Shape.Dimensions[i].Length != value.Shape.Dimensions[i].Length)
                        throw new Exception("Cannot assign incompatible arrays");

                UFunc.UFunc_Op_Inner_Unary<T, NumCIL.CopyOp<T>>(new NumCIL.CopyOp<T>(), value, ref lv);
            }
        }

        /// <summary>
        /// Gets a subview on the array
        /// </summary>
        /// <param name="ranges">The range get the view from</param>
        /// <returns>A view on the selected element</returns>
        public NdArray<T> this[params Range[] ranges]
        {
            get
            {
                if (ranges == null || ranges.Length == 0)
                    return this;

                NdArray<T> v = this;
                for(long i = 0; i < ranges.LongLength; i++)
                    v = v.Subview(ranges[i], i);

                //We reduce the last dimension if it only has one element
                while (ranges.LongLength == v.Shape.Dimensions.LongLength && v.Shape.Dimensions[v.Shape.Dimensions.LongLength - 1].Length == 1)
                {
                    long j = 0;
                    v = v.Reshape(new Shape(v.Shape.Dimensions.Where(x => j++ < v.Shape.Dimensions.LongLength - 1).ToArray(), v.Shape.Offset));
                }

                return v;
            }
            set
            {
                NdArray<T> lv = this[ranges];
                var broadcastShapes = Shape.ToBroadcastShapes(value.Shape, lv.Shape);
                UFunc.Apply<T, NumCIL.CopyOp<T>>(value.Reshape(broadcastShapes.Item1), lv.Reshape(broadcastShapes.Item2));
            }
        }

        /// <summary>
        /// Returns a flattened (1-d copy) of the current data view
        /// </summary>
        /// <returns>A flattened copy</returns>
        public NdArray<T> Flatten()
        {
            NdArray<T> cp = this.Clone();
            return new NdArray<T>(cp, new long[] { cp.Shape.Length });
        }

        /// <summary>
        /// Returns a copy of the underlying data, shaped as this view
        /// </summary>
        /// <returns>A copy of the view data</returns>
        public NdArray<T> Clone()
        {
            //TODO: Does not have a sane shape?
            return UFunc.Apply<T, CopyOp<T>>(this);
        }

        /// <summary>
        /// Sets all elements in the view to a specific value
        /// </summary>
        /// <param name="value"></param>
        public void Set(T value)
        {
            UFunc.Apply<T, GenerateOp<T>>(new GenerateOp<T>(value), this);
        }

        #region IEnumerable<NdArray<T>> Members

        /// <summary>
        /// Returns an enumerator that iterates through a collection.
        /// </summary>
        /// <returns>
        /// An <see cref="T:System.Collections.Generic.IEnumerator"/> object that can be used to iterate through the collection.
        /// </returns>
        public IEnumerator<NdArray<T>> GetEnumerator()
        {
            return new Utility.AnyEnumerator<NdArray<T>>(index => this[index], Shape.Dimensions[0].Length);
        }

        #endregion

        #region IEnumerable Members

        /// <summary>
        /// Returns an enumerator that iterates through a collection.
        /// </summary>
        /// <returns>
        /// An <see cref="T:System.Collections.IEnumerator"/> object that can be used to iterate through the collection.
        /// </returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        #endregion

        /// <summary>
        /// Returns a string representation of the data viewed by this NdArray
        /// </summary>
        /// <returns>A string representation of the data viewed by this NdArray</returns>
        public override string ToString()
        {
            return string.Format("NdArray<{0}>({1}): {2}", typeof(T).FullName, string.Join(", ", this.Shape.Dimensions.Select(x => x.Length.ToString()).ToArray()), this.AsString());
        }

        /// <summary>
        /// Returns the contents of this NdArray as a parseable string
        /// </summary>
        /// <param name="sb"></param>
        /// <returns></returns>
        public string AsString(StringBuilder sb = null)
        {
            sb = sb ?? new StringBuilder();

            if (this.Shape.Dimensions.LongLength == 1)
                sb.Append("[" + string.Join(", \n", this.Value.Select(x => x.ToString()).ToArray()) + "] ");
            else
                sb.Append("[" + string.Join(", \n", this.Select(x => x.AsString()).ToArray()) + "] ");

            return sb.ToString();
        }

        /// <summary>
        /// Flushes all pending operations on this array
        /// </summary>
        public void Flush()
        {
            if (m_data is ILazyAccessor<T>)
                ((ILazyAccessor<T>)m_data).Flush();
        }

        /// <summary>
        /// Extension to support unmanaged mapping
        /// </summary>
        public object Tag;

        /// <summary>
        /// Flag for debugging purposes
        /// </summary>
        public string Name;
    }
}

