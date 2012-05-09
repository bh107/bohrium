using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace NumCIL.Unsafe
{
    public static class Copy
    {
        public static void Memcpy(float[] target, IntPtr source, long count) { unsafe { fixed (float* t = target) { Inner.Memcpy(t, source.ToPointer(), 4, count); } } }
        public static void Memcpy(IntPtr target, float[] source, long count) { unsafe { fixed (float* s = source) { Inner.Memcpy(target.ToPointer(), s, 4, count); } } }
        public static void Memcpy(double[] target, IntPtr source, long count) { unsafe { fixed (double* t = target) { Inner.Memcpy(t, source.ToPointer(), 8, count); } } }
        public static void Memcpy(IntPtr target, double[] source, long count) { unsafe { fixed (double* s = source) { Inner.Memcpy(target.ToPointer(), s, 8, count); } } }

        private static unsafe class Inner
        {
            public static void Memcpy(void* target, void* source, int elsize, long count)
            {
                unsafe
                {
                    long bytes = elsize * count;
                    if (bytes % 8 == 0)
                    {
                        ulong* a = (ulong*)source;
                        ulong* b = (ulong*)target;
                        long els = bytes / 8;
                        for (long i = 0; i < els; i++)
                            b[i] = a[i];
                    }
                    else if (bytes % 4 == 0)
                    {
                        uint* a = (uint*)source;
                        uint* b = (uint*)target;
                        long els = bytes / 4;
                        for (long i = 0; i < els; i++)
                            b[i] = a[i];
                    }
                    else if (bytes % 2 == 0)
                    {
                        ushort* a = (ushort*)source;
                        ushort* b = (ushort*)target;
                        long els = bytes / 2;
                        for (long i = 0; i < els; i++)
                            b[i] = a[i];
                    }
                    else
                    {
                        byte* a = (byte*)source;
                        byte* b = (byte*)target;
                        long els = bytes;
                        for (long i = 0; i < els; i++)
                            b[i] = a[i];
                    }
                }
            }
        }
    }
}
