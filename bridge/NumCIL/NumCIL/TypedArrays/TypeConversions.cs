using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace NumCIL.Float
{
    using T = System.Single;
    using OutArray = NdArray;

    public partial class NdArray 
    {
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.Double
{
    using T = System.Double;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.Int8
{
    using T = System.SByte;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.Int16
{
    using T = System.Int16;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.Int32
{
    using T = System.Int32;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.Int64
{
    using T = System.Int64;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.UInt8
{
    using T = System.Byte;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.UInt16
{
    using T = System.UInt16;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.UInt32
{
    using T = System.UInt32;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt64 : IUnaryConvOp<T, ulong> { public ulong Op(T a) { return (ulong)a; } }
}

namespace NumCIL.UInt64
{
    using T = System.UInt64;
    using OutArray = NdArray;

    public partial class NdArray
    {
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
    }

    public struct ToFloat : IUnaryConvOp<T, float> { public float Op(T a) { return (float)a; } }
    public struct ToDouble : IUnaryConvOp<T, double> { public double Op(T a) { return (double)a; } }
    public struct ToInt8 : IUnaryConvOp<T, sbyte> { public sbyte Op(T a) { return (sbyte)a; } }
    public struct ToInt16 : IUnaryConvOp<T, short> { public short Op(T a) { return (short)a; } }
    public struct ToInt32 : IUnaryConvOp<T, int> { public int Op(T a) { return (int)a; } }
    public struct ToInt64 : IUnaryConvOp<T, long> { public long Op(T a) { return (long)a; } }
    public struct ToUInt8 : IUnaryConvOp<T, byte> { public byte Op(T a) { return (byte)a; } }
    public struct ToUInt16 : IUnaryConvOp<T, ushort> { public ushort Op(T a) { return (ushort)a; } }
    public struct ToUInt32 : IUnaryConvOp<T, uint> { public uint Op(T a) { return (uint)a; } }
}
