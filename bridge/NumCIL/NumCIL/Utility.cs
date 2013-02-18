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
using System.Runtime.InteropServices;

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

        /// <summary>
        /// Reads serialized elements from a file into a new array
        /// </summary>
        /// <typeparam name="T">The type of data to read</typeparam>
        /// <param name="file">The file to read from</param>
        /// <param name="elements">A maximum number of elements to read, a negative value means read everything</param>
        /// <returns>An array populated with data from the file</returns>
        public static T[] ReadArray<T>(string file, long elements = -1)
        {
            using (var fs = new System.IO.FileStream(file, System.IO.FileMode.Open, System.IO.FileAccess.Read))
                return ReadArray<T>(fs, elements);
        }

        /// <summary>
        /// Reads serialized elements from a file into a new array
        /// </summary>
        /// <typeparam name="T">The type of data to read</typeparam>
        /// <param name="fs">The stream to read from</param>
        /// <param name="elements">A maximum number of elements to read, a negative value means read everything</param>
        /// <returns>An array populated with data from the file</returns>
        public static T[] ReadArray<T>(System.IO.Stream fs, long elements = -1)
        {
            long streamlen = -1;
            int elsize = Marshal.SizeOf(typeof(T));
            try { streamlen = fs.Length; }
            catch { }

            if (elements < 0)
                elements = long.MaxValue;

            byte[] elbuf = new byte[1024 * 8 * elsize];

            if (streamlen == -1)
            {
                //We do not know the size in advance, so we allocate data in chunks
                long bytesread = 0;
                long elementsread = 0;
                int offset = 0;
                int a;
                List<T[]> r = new List<T[]>();
                while ((a = fs.Read(elbuf, offset, elbuf.Length - offset)) != 0)
                {
                    bytesread += a;
                    T[] p = new T[Math.Min(a / elsize, elements - elementsread)];
                    elementsread += p.LongLength;

                    //TODO: Free
                    GCHandle gh = new GCHandle();
                    try
                    {
                        gh = GCHandle.Alloc(p, GCHandleType.Pinned);
                        Marshal.Copy(elbuf, 0, gh.AddrOfPinnedObject(), p.Length * elsize);
                    }
                    finally
                    {
                        if (gh.IsAllocated)
                            gh.Free();
                    }

                    offset = a % elsize;
                    r.Add(p);

                    if (elements == elementsread)
                    {
                        offset = 0;
                        break;
                    }
                }

                if (offset != 0)
                    throw new System.IO.InvalidDataException(string.Format("Data size was {0}, but must be evenly divisible with {1}", bytesread, elsize));

                T[] result = new T[elementsread];
                long eloffset = 0;
                foreach (var e in r)
                {
                    Array.Copy(e, 0, result, eloffset, e.LongLength);
                    eloffset += e.LongLength;
                }

                if (elements != long.MaxValue && result.LongLength != elements)
                    throw new Exception("Internal error, read a wrong number of arguments");

                return result;
            }
            else
            {
                if (streamlen % elsize != 0)
                    throw new System.IO.InvalidDataException(string.Format("Data size is {0}, but must be evenly divisible with {1}", streamlen, elsize));

                //We can allocate the result in one go
                T[] result = new T[Math.Min(elements, streamlen / elsize)];

                GCHandle gh = new GCHandle();
                try
                {
                    gh = GCHandle.Alloc(result, GCHandleType.Pinned);
                    IntPtr curadr = gh.AddrOfPinnedObject();

                    long bytesread = 0;
                    long elementsread = 0;
                    int offset = 0;
                    int a;
                    while ((a = fs.Read(elbuf, offset, elbuf.Length - offset)) != 0)
                    {
                        int fes = (int)Math.Min(result.LongLength - elementsread, a / elsize);
                        elementsread += fes;

                        Marshal.Copy(elbuf, 0, curadr, fes * elsize);
                        offset = a % elsize;
                        bytesread += a;
                        curadr += fes * elsize;

                        if (elementsread == elements)
                        {
                            offset = 0;
                            break;
                        }
                    }

                    if (offset != 0 || (streamlen != bytesread && elements != long.MaxValue))
                        throw new System.IO.InvalidDataException(string.Format("Data size was {0}, but must be evenly divisible with {1}", bytesread, elsize));
                }
                finally
                {
                    if (gh.IsAllocated)
                        gh.Free();
                }

                if (elements != long.MaxValue && result.LongLength != elements)
                    throw new Exception("Internal error, read a wrong number of arguments");

                return result;
            }
        }

        /// <summary>
        /// Writes the input data to a file
        /// </summary>
        /// <typeparam name="T">The type of data to write</typeparam>
        /// <param name="data">The data to write</param>
        /// <param name="file">The file to write to</param>
        /// <param name="elements">The number of elements to write</param>
        /// <param name="offset">An optional offset into the array</param>
        public static void ToFile<T>(T[] data, string file, long elements = -1, long offset = 0)
        {
            using (var fs = new System.IO.FileStream(file, System.IO.FileMode.Create, System.IO.FileAccess.Write))
                ToFile<T>(data, fs, elements);
        }

        /// <summary>
        /// Writes the input data to a file
        /// </summary>
        /// <typeparam name="T">The type of data to write</typeparam>
        /// <param name="data">The data to write</param>
        /// <param name="fs">The stream to write to</param>
        /// <param name="elements">The number of elements to write</param>
        /// <param name="offset">An optional offset into the array</param>
        public static void ToFile<T>(T[] data, System.IO.Stream fs, long elements = -1, long offset = 0)
        {
            long elementstowrite = elements < 0 ? (data.LongLength - offset) : elements;
            if (data.LongLength < elements + offset)
                throw new ArgumentOutOfRangeException("data", string.Format("The size of the data is {0} but there should be {1} + {2} = {3} elements", data.LongLength, elements, offset, offset + elements));

            int elsize = Marshal.SizeOf(typeof(T));
            
            byte[] elbuf = new byte[1024 * 8 * elsize];
            GCHandle gh = new GCHandle();
            try
            {
                gh = GCHandle.Alloc(data, GCHandleType.Pinned);
                IntPtr curpos = gh.AddrOfPinnedObject();
                curpos += (int)(offset * elsize);

                while (elementstowrite > 0)
                {
                    int curels = (int)Math.Min(elementstowrite, elbuf.LongLength / elsize);
                    int bytes = curels * elsize;
                    Marshal.Copy(curpos, elbuf, 0, bytes);
                    fs.Write(elbuf, 0, bytes);
                    curpos += bytes;
                    elementstowrite -= curels;
                }
            }
            finally
            {
                if (gh.IsAllocated)
                    gh.Free();
            }
        }
    }
}
