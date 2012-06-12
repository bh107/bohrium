using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace NumCIL.Unsafe
{
    public class UnmanagedAccessorBase<T> : NumCIL.Generic.DefaultAccessor<T>, NumCIL.Generic.IUnmanagedDataAccessor<T>
    {
        protected IntPtr m_dataPtr = IntPtr.Zero;
        protected GCHandle m_handle;
        protected static readonly long DATA_ELEMENT_SIZE = Marshal.SizeOf(typeof(T));
        protected static readonly System.Reflection.MethodInfo COPYFROMMANAGED = Utility.GetCopyFromManaged<T>();
        protected static readonly System.Reflection.MethodInfo COPYTOMANAGED = Utility.GetCopyToManaged<T>();

        static UnmanagedAccessorBase()
        {
            if (COPYFROMMANAGED == null || COPYTOMANAGED == null)
                throw new NotSupportedException(string.Format("The type '{0}' is not supported by the unsafe implementation", typeof(T).FullName));
        }

        public UnmanagedAccessorBase(long size)
            : base(size)
        {
        }

        public UnmanagedAccessorBase(T[] data)
            : base(data)
        {
        }

        public override T this[long index]
        {
            get
            {
                throw new InvalidOperationException();
            }
            set
            {
                throw new InvalidOperationException();
            }
        }

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

        public override void Allocate()
        {
            if (m_data == null && m_dataPtr == IntPtr.Zero)
                m_dataPtr = Marshal.AllocHGlobal(new IntPtr(m_size * DATA_ELEMENT_SIZE));
        }

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

        public override bool IsAllocated
        {
            get
            {
                return m_dataPtr != IntPtr.Zero || base.IsAllocated;
            }
        }

        public bool CanAllocateArray
        {
            get { return m_size * DATA_ELEMENT_SIZE < int.MaxValue; }
        }

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
