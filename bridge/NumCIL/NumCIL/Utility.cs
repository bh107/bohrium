using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL
{
    /// <summary>
    /// Utility class with commonly used support methods
    /// </summary>
    public static class Utility
    {
        //TODO: Remove this class because it uses lambda functions which compile into virtual function calls, or modify the IL to generate a non-virtual function call type

        /// <summary>
        /// Simple helper class that can make any index-able class enumerable
        /// </summary>
        /// <typeparam name="T">The value type that is being enumerated</typeparam>
        public class AnyEnumerator<T> : IEnumerator<T>
        {
            /// <summary>
            /// The function used to extract data
            /// </summary>
            private Func<long, T> m_get;
            /// <summary>
            /// The length of the underlying data
            /// </summary>
            private long m_size;
            /// <summary>
            /// The current position in the enumeration
            /// </summary>
            private long m_index = -1;

            /// <summary>
            /// Constructs a new Enumerator
            /// </summary>
            /// <param name="get">The function used to extract an element from the underlying data</param>
            /// <param name="size">The length of the data</param>
            public AnyEnumerator(Func<long, T> get, long size)
            {
                m_get = get;
                m_size = size;
            }

            #region IEnumerator<T> Members

            /// <summary>
            /// Gets the current value
            /// </summary>
            public T Current
            {
                get { return m_get(m_index); }
            }

            #endregion

            #region IDisposable Members

            /// <summary>
            /// Disposes all unused resources
            /// </summary>
            public void Dispose()
            {
            }

            #endregion

            #region IEnumerator Members

            /// <summary>
            /// Gets the current element
            /// </summary>
            object System.Collections.IEnumerator.Current
            {
                get { return m_get(m_index); }
            }

            /// <summary>
            /// Advances the current element by a single position
            /// </summary>
            /// <returns>True if the index is within bounds, false otherwise</returns>
            public bool MoveNext()
            {
                m_index++;
                return m_index < m_size;
            }

            /// <summary>
            /// Sets the enumerator to point at the very first element in the collection
            /// </summary>
            public void Reset()
            {
                m_index = 0;
            }

            #endregion
        }

        /// <summary>
        /// An extension method that enables the use of LinQ on a plain IEnumerable, which is used extensively in Mono.Cecil
        /// </summary>
        /// <typeparam name="T">The type that the IEnumerable contains</typeparam>
        public class TypedEnumerable<T> : IEnumerable<T>
        {

            /// <summary>
            /// The collection that is bein enumerated
            /// </summary>
            private System.Collections.IEnumerable col;

            /// <summary>
            /// Initializes a new instance of the <see cref="TypedEnumerable&lt;T&gt;"/> class.
            /// </summary>
            /// <param name="col">The colletion to use</param>
            public TypedEnumerable(System.Collections.IEnumerable col) { this.col = col; }

            #region IEnumerable<Instruction> Members
            /// <summary>
            /// Gets the enumerator.
            /// </summary>
            /// <returns>An enumerator</returns>
            public IEnumerator<T> GetEnumerator() { return new TypedEnumerator(this.col.GetEnumerator()); }
            #endregion

            #region IEnumerable Members
            /// <summary>
            /// Returns an enumerator that iterates through a collection.
            /// </summary>
            /// <returns>
            /// An <see cref="T:System.Collections.IEnumerator"/> object that can be used to iterate through the collection.
            /// </returns>
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() { return this.col.GetEnumerator(); }
            #endregion

            /// <summary>
            /// A typed enumerator
            /// </summary>
            private class TypedEnumerator : IEnumerator<T>
            {
                /// <summary>
                /// The untyped enumerator that this instance is wrapping
                /// </summary>
                private System.Collections.IEnumerator m_base;
                /// <summary>
                /// Initializes a new instance of the <see cref="TypedEnumerable&lt;T&gt;.TypedEnumerator"/> class.
                /// </summary>
                /// <param name="base">The base enumerator</param>
                public TypedEnumerator(System.Collections.IEnumerator @base) { m_base = @base; }
                #region IEnumerator<T> Members
                /// <summary>
                /// Gets the current element
                /// </summary>
                public T Current { get { return (T)m_base.Current; } }
                #endregion

                #region IDisposable Members
                /// <summary>
                /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
                /// </summary>
                public void Dispose()
                {
                    if (m_base is IDisposable)
                        ((IDisposable)m_base).Dispose();
                }
                #endregion

                #region IEnumerator Members
                /// <summary>
                /// Gets the current element
                /// </summary>
                object System.Collections.IEnumerator.Current { get { return m_base.Current; } }
                /// <summary>
                /// Advances the enumerator to the next element of the collection.
                /// </summary>
                /// <returns>
                /// true if the enumerator was successfully advanced to the next element; false if the enumerator has passed the end of the collection.
                /// </returns>
                /// <exception cref="T:System.InvalidOperationException">The collection was modified after the enumerator was created. </exception>
                public bool MoveNext() { return m_base.MoveNext(); }
                /// <summary>
                /// Sets the enumerator to its initial position, which is before the first element in the collection.
                /// </summary>
                /// <exception cref="T:System.InvalidOperationException">The collection was modified after the enumerator was created. </exception>
                public void Reset() { m_base.MoveNext(); }
                #endregion
            }
        }

        /// <summary>
        /// Applies the specified action to all elements in the sequence
        /// </summary>
        /// <typeparam name="T">The type of data in the sequence</typeparam>
        /// <param name="seq">The sequence of data to work on</param>
        /// <param name="a">The action to perform on each element</param>
        public static void Apply<T>(this IEnumerable<T> seq, Action<T> a)
        {
            foreach (T n in seq)
                a(n);
        }
    }
}
