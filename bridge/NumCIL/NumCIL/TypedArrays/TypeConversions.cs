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
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float> 
    { 
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; } 
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong> 
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; } 
    }
}

namespace NumCIL.Double
{
    using T = System.Double;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}

namespace NumCIL.Int8
{
    using T = System.SByte;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}

namespace NumCIL.Int16
{
    using T = System.Int16;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}

namespace NumCIL.Int32
{
    using T = System.Int32;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}

namespace NumCIL.Int64
{
    using T = System.Int64;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}

namespace NumCIL.UInt8
{
    using T = System.Byte;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}

namespace NumCIL.UInt16
{
    using T = System.UInt16;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}

namespace NumCIL.UInt32
{
    using T = System.UInt32;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt64"/></returns>
        public static explicit operator NumCIL.UInt64.NdArray(OutArray a) { return UFunc.Apply<T, ulong, ToUInt64>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}

namespace NumCIL.UInt64
{
    using T = System.UInt64;
    using OutArray = NdArray;

    public partial class NdArray
    {
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Single"/></returns>
        public static explicit operator NumCIL.Float.NdArray(OutArray a) { return UFunc.Apply<T, float, ToFloat>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Double"/></returns>
        public static explicit operator NumCIL.Double.NdArray(OutArray a) { return UFunc.Apply<T, double, ToDouble>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.SByte"/></returns>
        public static explicit operator NumCIL.Int8.NdArray(OutArray a) { return UFunc.Apply<T, sbyte, ToInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int16"/></returns>
        public static explicit operator NumCIL.Int16.NdArray(OutArray a) { return UFunc.Apply<T, short, ToInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int32"/></returns>
        public static explicit operator NumCIL.Int32.NdArray(OutArray a) { return UFunc.Apply<T, int, ToInt32>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Int64"/></returns>
        public static explicit operator NumCIL.Int64.NdArray(OutArray a) { return UFunc.Apply<T, long, ToInt64>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.Byte"/></returns>
        public static explicit operator NumCIL.UInt8.NdArray(OutArray a) { return UFunc.Apply<T, byte, ToUInt8>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt16"/></returns>
        public static explicit operator NumCIL.UInt16.NdArray(OutArray a) { return UFunc.Apply<T, ushort, ToUInt16>(a); }
        /// <summary>
        /// Converts all elements in the input NdArray to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The input array</param>
        /// <returns>An NdArray of <see cref="System.UInt32"/></returns>
        public static explicit operator NumCIL.UInt32.NdArray(OutArray a) { return UFunc.Apply<T, uint, ToUInt32>(a); }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Single"/>
    /// </summary>
    public struct ToFloat : IUnaryConvOp<T, float>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Single"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public float Op(T a) { return (float)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Double"/>
    /// </summary>
    public struct ToDouble : IUnaryConvOp<T, double>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Double"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public double Op(T a) { return (double)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.SByte"/>
    /// </summary>
    public struct ToInt8 : IUnaryConvOp<T, sbyte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.SByte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public sbyte Op(T a) { return (sbyte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int16"/>
    /// </summary>
    public struct ToInt16 : IUnaryConvOp<T, short>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public short Op(T a) { return (short)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int32"/>
    /// </summary>
    public struct ToInt32 : IUnaryConvOp<T, int>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public int Op(T a) { return (int)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Int64"/>
    /// </summary>
    public struct ToInt64 : IUnaryConvOp<T, long>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Int64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public long Op(T a) { return (long)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.Byte"/>
    /// </summary>
    public struct ToUInt8 : IUnaryConvOp<T, byte>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.Byte"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public byte Op(T a) { return (byte)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt16"/>
    /// </summary>
    public struct ToUInt16 : IUnaryConvOp<T, ushort>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt16"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ushort Op(T a) { return (ushort)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt32"/>
    /// </summary>
    public struct ToUInt32 : IUnaryConvOp<T, uint>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt32"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public uint Op(T a) { return (uint)a; }
    }

    /// <summary>
    /// Operator for converting a value to <see cref="System.UInt64"/>
    /// </summary>
    public struct ToUInt64 : IUnaryConvOp<T, ulong>
    {
        /// <summary>
        /// Converts the input value to <see cref="System.UInt64"/>
        /// </summary>
        /// <param name="a">The value to convert</param>
        /// <returns>The converted valued</returns>
        public ulong Op(T a) { return (ulong)a; }
    }
}
