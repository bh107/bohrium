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
        public static void Memcpy(ulong[] target, IntPtr source, long count) { unsafe { fixed (ulong* t = target) { Inner.Memcpy(t, source.ToPointer(), 8, count); } } }
        public static void Memcpy(IntPtr target, ulong[] source, long count) { unsafe { fixed (ulong* s = source) { Inner.Memcpy(target.ToPointer(), s, 8, count); } } }
        public static void Memcpy(long[] target, IntPtr source, long count) { unsafe { fixed (long* t = target) { Inner.Memcpy(t, source.ToPointer(), 8, count); } } }
        public static void Memcpy(IntPtr target, long[] source, long count) { unsafe { fixed (long* s = source) { Inner.Memcpy(target.ToPointer(), s, 8, count); } } }
        public static void Memcpy(uint[] target, IntPtr source, long count) { unsafe { fixed (uint* t = target) { Inner.Memcpy(t, source.ToPointer(), 4, count); } } }
        public static void Memcpy(IntPtr target, uint[] source, long count) { unsafe { fixed (uint* s = source) { Inner.Memcpy(target.ToPointer(), s, 4, count); } } }
        public static void Memcpy(int[] target, IntPtr source, long count) { unsafe { fixed (int* t = target) { Inner.Memcpy(t, source.ToPointer(), 4, count); } } }
        public static void Memcpy(IntPtr target, int[] source, long count) { unsafe { fixed (int* s = source) { Inner.Memcpy(target.ToPointer(), s, 4, count); } } }
        public static void Memcpy(ushort[] target, IntPtr source, long count) { unsafe { fixed (ushort* t = target) { Inner.Memcpy(t, source.ToPointer(), 2, count); } } }
        public static void Memcpy(IntPtr target, ushort[] source, long count) { unsafe { fixed (ushort* s = source) { Inner.Memcpy(target.ToPointer(), s, 2, count); } } }
        public static void Memcpy(short[] target, IntPtr source, long count) { unsafe { fixed (short* t = target) { Inner.Memcpy(t, source.ToPointer(), 2, count); } } }
        public static void Memcpy(IntPtr target, short[] source, long count) { unsafe { fixed (short* s = source) { Inner.Memcpy(target.ToPointer(), s, 2, count); } } }
        public static void Memcpy(sbyte[] target, IntPtr source, long count) { unsafe { fixed (sbyte* t = target) { Inner.Memcpy(t, source.ToPointer(), 1, count); } } }
        public static void Memcpy(IntPtr target, sbyte[] source, long count) { unsafe { fixed (sbyte* s = source) { Inner.Memcpy(target.ToPointer(), s, 1, count); } } }
        public static void Memcpy(byte[] target, IntPtr source, long count) { unsafe { fixed (byte* t = target) { Inner.Memcpy(t, source.ToPointer(), 1, count); } } }
        public static void Memcpy(IntPtr target, byte[] source, long count) { unsafe { fixed (byte* s = source) { Inner.Memcpy(target.ToPointer(), s, 1, count); } } }

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
