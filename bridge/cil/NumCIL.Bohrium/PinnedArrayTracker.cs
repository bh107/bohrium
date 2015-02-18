using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

using bh_bool = System.Boolean;
using bh_int8 = System.SByte;
using bh_uint8 = System.Byte;
using bh_int16 = System.Int16;
using bh_uint16 = System.UInt16;
using bh_int32 = System.Int32;
using bh_uint32 = System.UInt32;
using bh_int64 = System.Int64;
using bh_uint64 = System.UInt64;
using bh_float32 = System.Single;
using bh_float64 = System.Double;
using bh_complex64 = NumCIL.Complex64.DataType;
using bh_complex128 = System.Numerics.Complex;

namespace NumCIL.Bohrium
{
    /// <summary>
    /// Utility class that keeps track of all pinned memory allocations,
    /// which is used for non-bohrium tracked arrays
    /// </summary>
    public static class PinnedArrayTracker
    {
		/// <summary>
		/// Pinned arrays that are to be unpinned
		/// </summary>
		private static List<GCHandle> _delayUnpins = new List<GCHandle>();

		/// <summary>
		/// Reference count for allocated data pointers
		/// </summary>
		private static Dictionary<IntPtr, long> _refCount = new Dictionary<IntPtr, long>();

		/// <summary>
		/// Handles for the pinned data entries
		/// </summary>
		private static Dictionary<IntPtr, GCHandle> _handles = new Dictionary<IntPtr, GCHandle>();

		/// <summary>
		/// List of allocated handles, used to detect multiple requests for pinning
		/// </summary>
		private static Dictionary<Array, IntPtr> _allocated = new Dictionary<Array, IntPtr>();

        /// <summary>
        /// The lock object that protects access to the pinner
        /// </summary>
        private static object _lock = new object();

		/// <summary>
		/// Environment variable that checks if the GC should be flushed before flushing instructions
		/// </summary>
		private static readonly bool BH_GC_FLUSH = "1".Equals(Environment.GetEnvironmentVariable("BH_GC_FLUSH"));

		/// <summary>
		/// Gets the execute lock object.
		/// </summary>
		/// <value>The execute lock object.</value>
		public static object ExecuteLock { get { return _lock; } }
    
		/// <summary>
		/// Pins an array and returns the data pointer
		/// </summary>
		/// <param name="item">The item to pin</param>
		/// <returns>The pinned data pointer</returns>
		public static IntPtr CreatePinnedArray(Array item)
		{
			lock (_lock)
			{
				IntPtr ptr;
				if (_allocated.TryGetValue(item, out ptr))
					return ptr;

				var handle = GCHandle.Alloc(item, GCHandleType.Pinned);
				ptr = handle.AddrOfPinnedObject();

				_handles.Add(ptr, handle);
				_refCount.Add(ptr, 0);
				_allocated.Add(item, ptr);

				return ptr;
			}
		}

		/// <summary>
		/// Determines if the data pointer is a managed data reference.
		/// </summary>
		/// <returns><c>true</c> if the data pointer is a managed data reference; otherwise, <c>false</c>.</returns>
		/// <param name="ptr">The data pointer</param>
		public static bool IsManagedData(IntPtr ptr)
		{
			if (ptr == IntPtr.Zero)
				return false;

			//lock (_lock)
			return _refCount.ContainsKey(ptr);
		}

		/// <summary>
		/// Incs the reference count.
		/// </summary>
		/// <param name="data">The data pointer</param>
		public static void IncReference(IntPtr data)
		{
			lock (_lock)
			{
				if (IsManagedData(data))
				{
					var d = _refCount[data];
					d++;
					_refCount[data] = d;
				}
			}
		}

		/// <summary>
		/// Decs the reference count.
		/// </summary>
		/// <param name="data">The data pointer</param>
		public static void DecReference(IntPtr data)
		{
			lock (_lock)
			{
				if (IsManagedData(data))
				{
					long rc = (_refCount[data] -= 1);
					if (rc == 0)
					{
						_refCount.Remove(data);
						_delayUnpins.Add(_handles[data]);
						_handles.Remove(data);

						//TODO: Slow
						foreach (var n in _allocated)
							if (n.Value == data)
							{
								_allocated.Remove(n.Key);
								break;
							}
					}
				}
			}
		}

        /// <summary>
        /// Flushes the pending queue and releases pinned arrays
        /// </summary>
        internal static void ReleaseInternal()
        {
			if (BH_GC_FLUSH)
			{
				GC.Collect();
				GC.WaitForPendingFinalizers();
			}

			lock (_lock)
			{
				// Execute all the operations
				PInvoke.bh_runtime_flush();

				/*if (_delayUnpins.Count != 0)
					Console.WriteLine("Unpinning {0} items", _delayUnpins.Count); */

				foreach (var h in _delayUnpins)
					h.Free();
				_delayUnpins.Clear();
			}
        }
			                
        /// <summary>
        /// Releases all pinned items
        /// </summary>
        public static void Release()
        {
            Utility.Flush();
        }
    }
}

