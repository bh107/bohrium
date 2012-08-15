#region Copyright
/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#endregion

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
    public class ViewPtrKeeper : IDisposable
    {
        /// <summary>
        /// Instance of the VEM that is used to dispose of the view
        /// </summary>
        protected static VEM VEM = NumCIL.cphVB.VEM.Instance;

        /// <summary>
        /// Flag to prevent double disposing
        /// </summary>
        private bool m_isDisposed = false;
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
        /// Gets a value indicating if the handle is allocated
        /// </summary>
        public bool HasHandle { get { return m_handle.IsAllocated; } }
        /// <summary>
        /// Gets a value indicating if this instance has been disposed
        /// </summary>
        public bool IsDisposed { get { return m_isDisposed; } }

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
            if (m_isDisposed)
                return;

            m_isDisposed = true;

            if (m_ptr != PInvoke.cphvb_array_ptr.Null)
            {
                if (m_handle.IsAllocated)
                {
                    VEM.ExecuteRelease(m_ptr, m_handle);
                }
                else if (m_ptr.Data != IntPtr.Zero && m_ptr.BaseArray == PInvoke.cphvb_array_ptr.Null)
                {
                    VEM.ExecuteRelease(
                        new PInvoke.cphvb_instruction(cphvb_opcode.CPHVB_FREE, m_ptr),
                        new PInvoke.cphvb_instruction(cphvb_opcode.CPHVB_DISCARD, m_ptr)
                    );
                }
                else
                {
                    VEM.ExecuteRelease(new PInvoke.cphvb_instruction(cphvb_opcode.CPHVB_DISCARD, m_ptr));
                }

                m_ptr = PInvoke.cphvb_array_ptr.Null;
            }
            else if (m_handle.IsAllocated)
            {
                m_handle.Free();
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
