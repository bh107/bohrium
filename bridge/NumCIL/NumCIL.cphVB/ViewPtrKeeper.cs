using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace NumCIL.cphVB
{
    /// <summary>
    /// This class that keeps a reference to an allocated cphvb_array_ptr,
    /// and is used to free allocated views on garbage collection
    /// </summary>
    internal class ViewPtrKeeper : IDisposable
    {
        /// <summary>
        /// Instance of the VEM that is used to dispose of the view
        /// </summary>
        protected static VEM VEM = NumCIL.cphVB.VEM.Instance;

        /// <summary>
        /// The view pointer
        /// </summary>
        private PInvoke.cphvb_array_ptr m_ptr;
        /// <summary>
        /// An optional GC handle for the views associated data
        /// </summary>
        private GCHandle m_handle;
        /// <summary>
        /// Gets the view pointer associated with this instance
        /// </summary>
        public PInvoke.cphvb_array_ptr Pointer { get { return m_ptr; } }

        /// <summary>
        /// Constructs a new instance guarding the given pointer
        /// </summary>
        /// <param name="p">The pointer to guard</param>
        public ViewPtrKeeper(PInvoke.cphvb_array_ptr p)
        {
            m_ptr = p;
        }

        /// <summary>
        /// Constructs a new instance guarding the given pointer
        /// </summary>
        /// <param name="p">The pointer to guard</param>
        /// <param name="handle">The associated handle</param>
        public ViewPtrKeeper(PInvoke.cphvb_array_ptr p, GCHandle handle)
        {
            System.Diagnostics.Debug.Assert(handle.IsAllocated);
            System.Diagnostics.Debug.Assert(p.Data == handle.AddrOfPinnedObject());

            m_ptr = p;
            m_handle = handle;
        }

        /// <summary>
        /// Cleans up associated data and frees the unmanaged resources relating to the view
        /// </summary>
        /// <param name="disposing">True if called from Dispose(), false if called fron the finalizer</param>
        public void Dispose(bool disposing)
        {
            bool doFree = true;
            if (m_handle.IsAllocated)
            {
                m_handle.Free();
                doFree = false;
            }

            if (m_ptr != PInvoke.cphvb_array_ptr.Null)
            {
                if (m_ptr.Data == IntPtr.Zero && m_ptr.BaseArray == PInvoke.cphvb_array_ptr.Null && doFree)
                    VEM.ExecuteRelease(new PInvoke.cphvb_instruction(cphvb_opcode.CPHVB_FREE, m_ptr));

                VEM.ExecuteRelease(new PInvoke.cphvb_instruction(cphvb_opcode.CPHVB_DISCARD, m_ptr));
                m_ptr = PInvoke.cphvb_array_ptr.Null;
            }

            if (disposing)
                GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Dispose all resources
        /// </summary>
        public void Dispose() { Dispose(true); }
        /// <summary>
        /// Finalize the object
        /// </summary>
        ~ViewPtrKeeper() { Dispose(false); }
    }
}
