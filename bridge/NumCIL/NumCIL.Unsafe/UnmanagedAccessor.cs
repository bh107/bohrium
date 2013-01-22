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
using System.Runtime.InteropServices;

namespace NumCIL.Unsafe
{
    /// <summary>
    /// Accessor for manually allocated memory, can be used to allocate memory blocks larger than the 2GB limit imposed by .NET
    /// </summary>
    /// <typeparam name="T">The datatype to represent</typeparam>
    public class UnmanagedAccessorBase<T> : NumCIL.Generic.DefaultAccessor<T>, NumCIL.Generic.IUnmanagedDataAccessor<T>
    {
        /// <summary>
        /// The pointer to allocated data
        /// </summary>
        protected IntPtr m_dataPtr = IntPtr.Zero;
        /// <summary>
        /// The GCHandle used for pinning the pointer
        /// </summary>
        protected GCHandle m_handle;
        /// <summary>
        /// Cached size of a single dataelement
        /// </summary>
        protected static readonly long DATA_ELEMENT_SIZE = Marshal.SizeOf(typeof(T));
        /// <summary>
        /// Cached handle for the memcopy method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo COPYFROMMANAGED = Utility.GetCopyFromManaged<T>();
        /// <summary>
        /// Cached handle for the memcopy method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo COPYTOMANAGED = Utility.GetCopyToManaged<T>();

        /// <summary>
        /// Static initializer, used to verify that the type can be used
        /// </summary>
        static UnmanagedAccessorBase()
        {
            if (COPYFROMMANAGED == null || COPYTOMANAGED == null)
                throw new NotSupportedException(string.Format("The type '{0}' is not supported by the unsafe implementation", typeof(T).FullName));
        }

        /// <summary>
        /// Creates a new accessor for the given size
        /// </summary>
        /// <param name="size">The number of elements to represent</param>
        public UnmanagedAccessorBase(long size)
            : base(size)
        {
        }

        /// <summary>
        /// Creates a new accessor for the allocated array
        /// </summary>
        /// <param name="data">The allocated data</param>
        public UnmanagedAccessorBase(T[] data)
            : base(data)
        {
        }

        /// <summary>
        /// Gets or sets an element in the array, not supported
        /// </summary>
        /// <param name="index">The index of the element to access</param>
        /// <returns></returns>
        public override T this[long index]
        {
            get
            {
                if (index < 0 || index >= m_size)
                    throw new ArgumentOutOfRangeException("index");
                Allocate();
                if (m_data != null)
                {
                    return m_data[index];
                }
                else
                {
                    if (m_dataPtr == IntPtr.Zero)
                        Allocate();

                    T[] tmp = new T[1];
                    COPYTOMANAGED.Invoke(null, new object[] { tmp, m_dataPtr, 1 });
                    return tmp[0];
                }
            }
            set
            {
                if (index < 0 || index >= m_size)
                    throw new ArgumentOutOfRangeException("index");

                if (m_data != null)
                {
                    m_data[index] = value;
                }
                else
                {
                    if (m_dataPtr == IntPtr.Zero)
                        Allocate();
                    COPYTOMANAGED.Invoke(null, new object[] { m_dataPtr, new T[] { value }, 1 });
                }
            }
        }

        /// <summary>
        /// Returns the data as an array
        /// </summary>
        /// <returns>The data as an array</returns>
        public override T[] AsArray()
        {
            if (m_dataPtr != IntPtr.Zero && m_data == null)
            {
                base.Allocate();
                COPYTOMANAGED.Invoke(null, new object[] { base.AsArray(), m_dataPtr, m_size });
                m_dataPtr = IntPtr.Zero;
                if (m_handle.IsAllocated)
                    m_handle.Free();
            }

            base.Allocate();
            return base.AsArray();
        }

        /// <summary>
        /// Allocates data
        /// </summary>
        public override void Allocate()
        {
            if (m_data == null && m_dataPtr == IntPtr.Zero)
                m_dataPtr = Marshal.AllocHGlobal(new IntPtr(m_size * DATA_ELEMENT_SIZE));
        }

        /// <summary>
        /// Gets a pointer to the data
        /// </summary>
        public IntPtr Pointer
        {
            get 
            {
                Allocate();
                if (m_dataPtr == IntPtr.Zero)
                {
                    m_handle = GCHandle.Alloc(m_data, GCHandleType.Pinned);
                    m_dataPtr = m_handle.AddrOfPinnedObject();
                }

                return m_dataPtr;
            }
        }

        /// <summary>
        /// Returns a value indicating if the data is allocated
        /// </summary>
        public override bool IsAllocated
        {
            get
            {
                return m_dataPtr != IntPtr.Zero || base.IsAllocated;
            }
        }

        /// <summary>
        /// Returns a value indicating if it is possible to allocate the data as an array
        /// </summary>
        public bool CanAllocateArray
        {
            get { return m_size * DATA_ELEMENT_SIZE < int.MaxValue; }
        }

        /// <summary>
        /// Finalizer for releasing resources
        /// </summary>
        ~UnmanagedAccessorBase()
        {
            if (m_data != null)
                m_data = null;

            if (m_handle.IsAllocated)
            {
                m_dataPtr = IntPtr.Zero;
                m_handle.Free();
            }
            else
            {
                Marshal.FreeHGlobal(m_dataPtr);
                m_dataPtr = IntPtr.Zero;
            }

            m_size = -1;
        }
    }
}
