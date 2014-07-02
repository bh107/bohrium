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

namespace NumCIL.Bohrium2
{
    /// <summary>
    /// Utility class that keeps track of all pinned memory allocations,
    /// which is used for non-bohrium tracked arrays
    /// </summary>
    public static class PinnedArrayTracker
    {
        /// <summary>
        /// The lookup table with all pinned items
        /// </summary>
        private static Dictionary<Array, Tuple<IntPtr, GCHandle, IDisposable>> _allocations = new Dictionary<Array, Tuple<IntPtr, GCHandle, IDisposable>>();

        /// <summary>
        /// The lock object that protects access to the pinner
        /// </summary>
        private static object _lock = new object();
    
        /// <summary>
        /// Pins the array.
        /// </summary>
        /// <param name="item">Item.</param>
        /// <typeparam name="T">The 1st type parameter.</typeparam>
        public static IntPtr PinArray<T>(T[] item)
        {
            lock (_lock)
            {
                Tuple<IntPtr, GCHandle, IDisposable> r;
                if (_allocations.TryGetValue(item, out r))
                    return r.Item1;

                var handle = GCHandle.Alloc(item, GCHandleType.Pinned);
                if (typeof(T) == typeof(bh_bool))
                {
                    var p = new PInvoke.bh_base_bool8_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_uint8))
                {
                    var p = new PInvoke.bh_base_uint8_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_int8))
                {
                    var p = new PInvoke.bh_base_int8_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_int16))
                {
                    var p = new PInvoke.bh_base_int16_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_uint16))
                {
                    var p = new PInvoke.bh_base_uint16_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_int32))
                {
                    var p = new PInvoke.bh_base_int32_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_uint32))
                {
                    var p = new PInvoke.bh_base_uint32_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_int64))
                {
                    var p = new PInvoke.bh_base_int64_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_uint64))
                {
                    var p = new PInvoke.bh_base_uint64_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_float32))
                {
                    var p = new PInvoke.bh_base_float32_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_float64))
                {
                    var p = new PInvoke.bh_base_float64_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_complex64))
                {
                    var p = new PInvoke.bh_base_complex64_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else if (typeof(T) == typeof(bh_complex128))
                {
                    var p = new PInvoke.bh_base_complex128_p(handle.AddrOfPinnedObject(), item.Length);
                    r = new Tuple<IntPtr, GCHandle, IDisposable>(
                        p.ToPointer(),
                        handle,
                        p
                    );
                }
                else
                {
                    handle.Free();
                    throw new Exception("Unexpected data type: " + typeof(T).FullName);
                }
                
                _allocations[item] = r;
                return r.Item1;
            }
        }
        
        /// <summary>
        /// Gets a value indicating if there are pinned entries
        /// </summary>
        /// <value><c>true</c> there are pinned entries; otherwise, <c>false</c>.</value>
        public static bool HasEntries { get { return _allocations.Count != 0; } }
        
        /// <summary>
        /// Releases all pinned items
        /// </summary>
        public static void Release()
        {
            lock (_lock)
            {
                // Ensure we have nothing that depends on this
                Utility.Flush();
                
                foreach (var h in _allocations.Values)
                {
                    h.Item3.Dispose();
                    h.Item2.Free();
                }
                
                _allocations.Clear();
            }
        }
    }
}

