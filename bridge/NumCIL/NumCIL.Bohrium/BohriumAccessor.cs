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
using NumCIL.Generic;
using System.Runtime.InteropServices;

namespace NumCIL.Bohrium
{
    /// <summary>
    /// Basic factory for creating Bohrium accessors
    /// </summary>
    /// <typeparam name="T">The type of data kept in the underlying array</typeparam>
    public class BohriumAccessorFactory<T> : NumCIL.Generic.IAccessorFactory<T>
    {
        /// <summary>
        /// Creates a new accessor for a data chunk of the given size
        /// </summary>
        /// <param name="size">The size of the array</param>
        /// <returns>An accessor</returns>
        public IDataAccessor<T> Create(long size) { return size == 1 ? new BohriumAccessor<T>(new T[1]) : new BohriumAccessor<T>(size); }
        /// <summary>
        /// Creates a new accessor for a preallocated array
        /// </summary>
        /// <param name="data">The data to wrap</param>
        /// <returns>An accessor</returns>
        public IDataAccessor<T> Create(T[] data) { return new BohriumAccessor<T>(data); }
    }
    
    /// <summary>
    /// Code to map from NumCIL operations to Bohrium operations
    /// </summary>
    public class OpCodeMapper
    {
        /// <summary>
        /// Lookup table with mapping from NumCIL operation name to Bohrium opcode
        /// </summary>
        private static Dictionary<bh_opcode, string> _opcode_func_name;

        /// <summary>
        /// Static initializer, builds mapping table between the Bohrium opcodes.
        /// and the corresponding names of the operations in NumCIL
        /// </summary>
        static OpCodeMapper()
        {
            _opcode_func_name = new Dictionary<bh_opcode, string>();

            _opcode_func_name.Add(bh_opcode.BH_ADD, "Add");
            _opcode_func_name.Add(bh_opcode.BH_SUBTRACT, "Sub");
            _opcode_func_name.Add(bh_opcode.BH_MULTIPLY, "Mul");
            _opcode_func_name.Add(bh_opcode.BH_DIVIDE, "Div");
            _opcode_func_name.Add(bh_opcode.BH_MOD, "Mod");
            _opcode_func_name.Add(bh_opcode.BH_MAXIMUM, "Max");
            _opcode_func_name.Add(bh_opcode.BH_MINIMUM, "Min");

            //These two are not found in Bohrium, but are emulated with ADD and SUB
            //_opcode_func_name.Add(bh_opcode.BH_INCREMENT, "Inc");
            //_opcode_func_name.Add(bh_opcode.BH_DECREMENT, "Dec");
            _opcode_func_name.Add(bh_opcode.BH_FLOOR, "Floor");
            _opcode_func_name.Add(bh_opcode.BH_CEIL, "Ceiling");
            _opcode_func_name.Add(bh_opcode.BH_RINT, "Round");

            _opcode_func_name.Add(bh_opcode.BH_ABSOLUTE, "Abs");
            _opcode_func_name.Add(bh_opcode.BH_SQRT, "Sqrt");
            _opcode_func_name.Add(bh_opcode.BH_EXP, "Exp");
            _opcode_func_name.Add(bh_opcode.BH_LOG, "Log");
            _opcode_func_name.Add(bh_opcode.BH_LOG10, "Log10");
            _opcode_func_name.Add(bh_opcode.BH_POWER, "Pow");

            _opcode_func_name.Add(bh_opcode.BH_COS, "Cos");
            _opcode_func_name.Add(bh_opcode.BH_SIN, "Sin");
            _opcode_func_name.Add(bh_opcode.BH_TAN, "Tan");
            _opcode_func_name.Add(bh_opcode.BH_ARCCOS, "Acos");
            _opcode_func_name.Add(bh_opcode.BH_ARCSIN, "Asin");
            _opcode_func_name.Add(bh_opcode.BH_ARCTAN, "Atan");
            _opcode_func_name.Add(bh_opcode.BH_COSH, "Cosh");
            _opcode_func_name.Add(bh_opcode.BH_SINH, "Sinh");
            _opcode_func_name.Add(bh_opcode.BH_TANH, "Tanh");

            _opcode_func_name.Add(bh_opcode.BH_LOGICAL_NOT, "Not");
            _opcode_func_name.Add(bh_opcode.BH_INVERT, "Invert");
        }

        /// <summary>
        /// Gets the specialized operation for given operand name
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>The specialized operation for the given operand or null</returns>
        public static Type GetOp<T>(string name)
        {
            Type basic = GetBasicClass<T>();
            try { return basic.Assembly.GetType(basic.Namespace + "." + name); }
            catch { return null; }
        }

        /// <summary>
        /// Returns the specialized NdArray class given the input element type
        /// </summary>
        /// <typeparam name="T">The input element type</typeparam>
        /// <returns>The type of the specialized NdArray</returns>
        protected static Type GetBasicClass<T>()
        {
            if (typeof(T) == typeof(sbyte))
                return typeof(NumCIL.Int8.NdArray);
            else if (typeof(T) == typeof(short))
                return typeof(NumCIL.Int16.NdArray);
            else if (typeof(T) == typeof(int))
                return typeof(NumCIL.Int32.NdArray);
            else if (typeof(T) == typeof(long))
                return typeof(NumCIL.Int64.NdArray);
            else if (typeof(T) == typeof(byte))
                return typeof(NumCIL.UInt8.NdArray);
            else if (typeof(T) == typeof(ushort))
                return typeof(NumCIL.UInt16.NdArray);
            else if (typeof(T) == typeof(uint))
                return typeof(NumCIL.UInt32.NdArray);
            else if (typeof(T) == typeof(ulong))
                return typeof(NumCIL.UInt64.NdArray);
            else if (typeof(T) == typeof(float))
                return typeof(NumCIL.Float.NdArray);
            else if (typeof(T) == typeof(double))
                return typeof(NumCIL.Double.NdArray);
            else if (typeof(T) == typeof(double))
                return typeof(NumCIL.Double.NdArray);
            else if (typeof(T) == typeof(bool))
                return typeof(NumCIL.Boolean.NdArray);
            else if (typeof(T) == typeof(NumCIL.Complex64.DataType))
                return typeof(NumCIL.Complex64.NdArray);
            else if (typeof(T) == typeof(System.Numerics.Complex))
                return typeof(NumCIL.Complex128.NdArray);
            else
                throw new Exception("Unexpected type: " + (typeof(T)).FullName);            
        }

        /// <summary>
        /// Helper function to get the opcode mapping table for the current type
        /// </summary>
        /// <returns>A mapping between the type used for this executor and the Bohrium opcodes</returns>
        public static Dictionary<Type, bh_opcode> CreateOpCodeMap<T>()
        {
            Dictionary<Type, bh_opcode> res = new Dictionary<Type, bh_opcode>();

            Type basic = GetBasicClass<T>();
            Dictionary<bh_opcode, string> opcodenames = new Dictionary<bh_opcode, string>(_opcode_func_name);

            if (typeof(T) == typeof(bool))
            {
                opcodenames.Add(bh_opcode.BH_LOGICAL_AND, "And");
                opcodenames.Add(bh_opcode.BH_LOGICAL_OR, "Or");
                opcodenames.Add(bh_opcode.BH_LOGICAL_XOR, "Xor");
            }
            else
            {
                opcodenames.Add(bh_opcode.BH_BITWISE_AND, "And");
                opcodenames.Add(bh_opcode.BH_BITWISE_OR, "Or");
                opcodenames.Add(bh_opcode.BH_BITWISE_XOR, "Xor");
            }

            foreach (var e in opcodenames)
            {
                try 
                {
                    Type t = basic.Assembly.GetType(basic.Namespace + "." + e.Value);
                    if (t != null)
                        res[t] = e.Key; 
                }
                catch { }
            }

            res[typeof(NumCIL.CopyOp<T>)] = bh_opcode.BH_IDENTITY;
            res[typeof(NumCIL.GenerateOp<T>)] = bh_opcode.BH_IDENTITY;
			res[typeof(NumCIL.UFunc.LazyReduceOperation<T>)] = bh_opcode.BH_USERFUNC;
			res[typeof(NumCIL.UFunc.LazyAggregateOperation<T>)] = bh_opcode.BH_USERFUNC;
			if (VEM.Instance.SupportsRandom)
			{
                res[typeof(NumCIL.Generic.IRandomGeneratorOp<T>)] = bh_opcode.BH_USERFUNC;
				try { res[basic.Assembly.GetType("NumCIL.Generic.RandomGeneratorOp" + (typeof(T) == typeof(float) ? "Float" : typeof(T).Name))] = bh_opcode.BH_USERFUNC; }
				catch {}
			}
            if (VEM.Instance.SupportsMatmul)
                res[typeof(NumCIL.UFunc.LazyMatmulOperation<T>)] = bh_opcode.BH_USERFUNC;


            if (typeof(T) == typeof(NumCIL.Complex64.DataType))
            {
                res[typeof(NumCIL.Complex64.ToComplex)] = bh_opcode.BH_IDENTITY;
            }
            else if (typeof(T) == typeof(System.Numerics.Complex))
            {
                res[typeof(NumCIL.Complex128.ToComplex)] = bh_opcode.BH_IDENTITY;
            }
            else
            {
                foreach (var e in new string[] {"Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64", "Float", "Double"})
                {
                    try 
                    {
                        Type t = basic.Assembly.GetType(basic.Namespace + ".To" + e);
                        if (t != null)
                            res[t] = bh_opcode.BH_IDENTITY; 
                    }
                    catch { }
                }
            }

            if (typeof(T) == typeof(bool))
            {
                Dictionary<bh_opcode, string> logicalnames = new Dictionary<bh_opcode, string>();
                logicalnames.Add(bh_opcode.BH_EQUAL, "Equal");
                logicalnames.Add(bh_opcode.BH_NOT_EQUAL, "NotEqual");
                logicalnames.Add(bh_opcode.BH_GREATER, "GreaterThan");
                logicalnames.Add(bh_opcode.BH_LESS, "LessThan");
                logicalnames.Add(bh_opcode.BH_GREATER_EQUAL, "GreaterThanOrEqual");
                logicalnames.Add(bh_opcode.BH_LESS_EQUAL, "LessThanOrEqual");

                foreach (var type in new Type[] { typeof(NumCIL.Int8.NdArray), typeof(NumCIL.UInt8.NdArray), typeof(NumCIL.Int16.NdArray), typeof(NumCIL.UInt16.NdArray), typeof(NumCIL.Int32.NdArray), typeof(NumCIL.UInt32.NdArray), typeof(NumCIL.Int64.NdArray), typeof(NumCIL.UInt64.NdArray), typeof(NumCIL.Float.NdArray), typeof(NumCIL.Double.NdArray), typeof(NumCIL.Complex64.NdArray), typeof(NumCIL.Complex128.NdArray) })
                {
                    foreach (var e in logicalnames)
                    {
                        Type t = basic.Assembly.GetType(type.Namespace + "." + e.Value);
                        if (t != null)
                            res[t] = e.Key;

                    }
                }
            }
            else
            {

                try
                {
                    Type basicBool = GetBasicClass<bool>();
                    string boolConvOpName = basicBool.Namespace + ".To" + basic.Namespace.Substring("NumCIL.".Length);
                    Type t = basicBool.Assembly.GetType(boolConvOpName);
                    if (t != null)
                        res[t] = bh_opcode.BH_IDENTITY;
                }
                catch
                {
                }
            }


            return res;
        }
    }

    /// <summary>
    /// Basic accessor for a Bohrium array
    /// </summary>
    /// <typeparam name="T">The type of data kept in the underlying array</typeparam>
    public class BohriumAccessor<T> : NumCIL.Generic.LazyAccessor<T>, IDisposable, IUnmanagedDataAccessor<T>
    {
        /// <summary>
        /// Lock that prevents multithreaded access to the Bohrium data
        /// </summary>
        private readonly object m_lock = new object();

        /// <summary>
        /// Instance of the VEM that is used
        /// </summary>
        protected static VEM VEM = NumCIL.Bohrium.VEM.Instance;

        /// <summary>
        /// The maximum number of instructions to queue
        /// </summary>
        protected static readonly long HIGH_WATER_MARK = 4000;

        /// <summary>
        /// Local copy of the type, to avoid lookups in the VEM dictionary
        /// </summary>
        protected static readonly PInvoke.bh_type BH_TYPE = VEM.MapType(typeof(T));

        /// <summary>
        /// The size of the data element in native code
        /// </summary>
        protected static readonly int NATIVE_ELEMENT_SIZE = Marshal.SizeOf(typeof(T));

        /// <summary>
        /// A lookup table that maps NumCIL operation types to Bohrium opcodes
        /// </summary>
        protected static Dictionary<Type, bh_opcode> OpcodeMap = OpCodeMapper.CreateOpCodeMap<T>();

        /// <summary>
        /// Gets the type for the Add operation
        /// </summary>
        protected static readonly Type AddOp = OpCodeMapper.GetOp<T>("Add");

        /// <summary>
        /// Gets the type for the Sub operation
        /// </summary>
        protected static readonly Type SubOp = OpCodeMapper.GetOp<T>("Sub");

		/// <summary>
		/// Gets the the generic template used to create conversion instructions
		/// </summary>
		protected static readonly System.Reflection.MethodInfo VEMConversionMethod = typeof(VEM).GetMethod("CreateConversionInstruction");

        /// <summary>
        /// Constructs a new data accessor for the given size
        /// </summary>
        /// <param name="size">The size of the data</param>
        public BohriumAccessor(long size) : base(size) { }

        /// <summary>
        /// Constructs a new data accessor for a pre-allocated block of storage
        /// </summary>
        /// <param name="data"></param>
        public BohriumAccessor(T[] data) : base(data) { }

        /// <summary>
        /// A pointer to the base-array view structure
        /// </summary>
        protected ViewPtrKeeper m_externalData = null;

        /// <summary>
        /// Ensures that local data is synced
        /// </summary>
        private void EnsureSynced()
        {
            this.Flush();
            if (m_data == null && m_externalData == null)
                base.Allocate();

            if (m_externalData != null)
                VEM.Execute(new PInvoke.bh_instruction(bh_opcode.BH_SYNC, m_externalData.Pointer));
        }

        /// <summary>
        /// Flushes all pending instructions, allocates data region and returns the contents as an array
        /// </summary>
        /// <returns></returns>
        public override T[] AsArray()
        {
            MakeDataManaged();
            return m_data;
        }

        /// <summary>
        /// Accesses an element, this method ensures that all pending instructions are flushed
        /// </summary>
        /// <param name="index">The element to accesss</param>
        /// <returns>The element at the specified address</returns>
        public override T this[long index]
        {
            get
            {
                if (index < 0 || index >= m_size)
                    throw new ArgumentOutOfRangeException("index");

                this.EnsureSynced();
                if (m_data != null)
                    return m_data[index];
                else
                {
                    if (m_externalData.Pointer.Data == IntPtr.Zero)
                    {
                        throw new Exception("Data not yet allocated?");
                    }

                    IntPtr ptr = new IntPtr(m_externalData.Pointer.Data.ToInt64() + (index * NATIVE_ELEMENT_SIZE));
                    if (typeof(T) == typeof(float))
                    {
                        T[] tmp = new T[1];
                        if (!NumCIL.UnsafeAPI.CopyFromIntPtr(ptr, tmp, 1))
                            Marshal.Copy(ptr, (float[])(object)tmp, 0, 1);
                        return tmp[0];
                    }
                    else if (typeof(T) == typeof(double))
                    {
                        T[] tmp = new T[1];
                        if (!NumCIL.UnsafeAPI.CopyFromIntPtr(ptr, tmp, 1))
                            Marshal.Copy(ptr, (double[])(object)tmp, 0, 1);
                        return tmp[0];
                    }
                    else if (typeof(T) == typeof(sbyte))
                        return (T)(object)(sbyte)Marshal.ReadByte(ptr);
                    else if (typeof(T) == typeof(short))
                        return (T)(object)(short)Marshal.ReadInt16(ptr);
                    else if (typeof(T) == typeof(int))
                        return (T)(object)(int)Marshal.ReadInt32(ptr);
                    else if (typeof(T) == typeof(long))
                        return (T)(object)(long)Marshal.ReadInt64(ptr);
                    else if (typeof(T) == typeof(byte))
                        return (T)(object)(byte)Marshal.ReadByte(ptr);
                    else if (typeof(T) == typeof(ushort))
                        return (T)(object)(ushort)Marshal.ReadInt16(ptr);
                    else if (typeof(T) == typeof(uint))
                        return (T)(object)(sbyte)Marshal.ReadInt32(ptr);
                    else if (typeof(T) == typeof(ulong))
                        return (T)(object)(sbyte)Marshal.ReadInt64(ptr);
                    else if (typeof(T) == typeof(NumCIL.Complex64.DataType))
                    {
                        float[] tmp = new float[2];
                        if (!NumCIL.UnsafeAPI.CopyFromIntPtr(ptr, tmp, 2))
                            Marshal.Copy(ptr, tmp, 0, 2);
                        return (T)(object)new NumCIL.Complex64.DataType(tmp[0], tmp[1]);
                    }
                    else if (typeof(T) == typeof(System.Numerics.Complex))
                    {
                        double[] tmp = new double[2];
                        if (!NumCIL.UnsafeAPI.CopyFromIntPtr(ptr, tmp, 2))
                            Marshal.Copy(ptr, tmp, 0, 2);
                        return (T)(object)new System.Numerics.Complex(tmp[0], tmp[1]);
                    }
                    else
                        throw new BohriumException(string.Format("Unexpected data type: {0}", typeof(T).FullName));
                }
            }
            set
            {
                MakeDataManaged();
                m_data[index] = value;
            }
        }

        /// <summary>
        /// Ensures that data is managed
        /// </summary>
        private void MakeDataManaged()
        {
            //TODO: Reconsider if this should be handled in another way than with a lock here
            lock (m_lock)
            {
                EnsureSynced();
                if (m_data != null && m_externalData == null)
                    return;

                if (m_data == null)
                {
                    base.Allocate();

                    IntPtr actualData = m_externalData.Pointer.Data;
                    if (actualData == IntPtr.Zero)
                    {
                        //The array is "empty" which will be zeroes in NumCIL
                    }
                    else
                    {
                        //Then copy the data into the local buffer
                        if (!NumCIL.UnsafeAPI.CopyFromIntPtr<T>(actualData, m_data))
                        {
                            if (m_size > int.MaxValue)
                                throw new OverflowException();

                            if (typeof(T) == typeof(float))
                                Marshal.Copy(actualData, (float[])(object)m_data, 0, (int)m_size);
                            else if (typeof(T) == typeof(double))
                                Marshal.Copy(actualData, (double[])(object)m_data, 0, (int)m_size);
                            else if (typeof(T) == typeof(sbyte))
                            {
                                sbyte[] xref = (sbyte[])(object)m_data;
                                if (m_size > int.MaxValue)
                                {
                                    IntPtr xptr = actualData;
                                    for (long i = 0; i < m_size; i++)
                                    {
                                        xref[i] = (sbyte)Marshal.ReadByte(xptr);
                                        xptr = IntPtr.Add(xptr, NATIVE_ELEMENT_SIZE);
                                    }
                                }
                                else
                                {
                                    for (int i = 0; i < m_size; i++)
                                        xref[i] = (sbyte)Marshal.ReadByte(actualData);
                                }
                            }
                            else if (typeof(T) == typeof(short))
                                Marshal.Copy(actualData, (short[])(object)m_data, 0, (int)m_size);
                            else if (typeof(T) == typeof(int))
                                Marshal.Copy(actualData, (int[])(object)m_data, 0, (int)m_size);
                            else if (typeof(T) == typeof(long))
                                Marshal.Copy(actualData, (long[])(object)m_data, 0, (int)m_size);
                            else if (typeof(T) == typeof(byte))
                                Marshal.Copy(actualData, (byte[])(object)m_data, 0, (int)m_size);
                            else if (typeof(T) == typeof(ushort))
                            {
                                ushort[] xref = (ushort[])(object)m_data;
                                if (m_size > int.MaxValue)
                                {
                                    IntPtr xptr = actualData;
                                    for (long i = 0; i < m_size; i++)
                                    {
                                        xref[i] = (ushort)Marshal.ReadInt16(xptr);
                                        xptr = IntPtr.Add(xptr, NATIVE_ELEMENT_SIZE);
                                    }
                                }
                                else
                                {
                                    for (int i = 0; i < m_size; i++)
                                        xref[i] = (ushort)Marshal.ReadInt16(actualData);
                                }
                            }
                            else if (typeof(T) == typeof(uint))
                            {
                                uint[] xref = (uint[])(object)m_data;
                                if (m_size > int.MaxValue)
                                {
                                    IntPtr xptr = actualData;
                                    for (long i = 0; i < m_size; i++)
                                    {
                                        xref[i] = (uint)Marshal.ReadInt32(xptr);
                                        xptr = IntPtr.Add(xptr, NATIVE_ELEMENT_SIZE);
                                    }
                                }
                                else
                                {
                                    for (int i = 0; i < m_size; i++)
                                        xref[i] = (uint)Marshal.ReadInt32(actualData);
                                }
                            }
                            else if (typeof(T) == typeof(ulong))
                            {
                                ulong[] xref = (ulong[])(object)m_data;
                                if (m_size > int.MaxValue)
                                {
                                    IntPtr xptr = actualData;
                                    for (long i = 0; i < m_size; i++)
                                    {
                                        xref[i] = (ulong)Marshal.ReadInt64(xptr);
                                        xptr = IntPtr.Add(xptr, NATIVE_ELEMENT_SIZE);
                                    }
                                }
                                else
                                {
                                    for (int i = 0; i < m_size; i++)
                                        xref[i] = (ulong)Marshal.ReadInt64(actualData);
                                }
                            }
                            else if (typeof(T) == typeof(NumCIL.Complex64.DataType))
                            {
                                NumCIL.Complex64.DataType[] xref = (NumCIL.Complex64.DataType[])(object)m_data;
                                float[] tmp = new float[2];
                                IntPtr xptr = actualData;
                                for (long i = 0; i < m_size; i++)
                                {
                                    Marshal.Copy(xptr, tmp, 0, (int)2);
                                    xref[i] = new NumCIL.Complex64.DataType(tmp[0], tmp[1]);
                                    xptr = IntPtr.Add(xptr, NATIVE_ELEMENT_SIZE);
                                }
                            }
                            else if (typeof(T) == typeof(System.Numerics.Complex))
                            {
                                System.Numerics.Complex[] xref = (System.Numerics.Complex[])(object)m_data;
                                double[] tmp = new double[2];
                                IntPtr xptr = actualData;
                                for (long i = 0; i < m_size; i++)
                                {
                                    Marshal.Copy(xptr, tmp, 0, (int)2);
                                    xref[i] = new System.Numerics.Complex(tmp[0], tmp[1]);
                                    xptr = IntPtr.Add(xptr, NATIVE_ELEMENT_SIZE);
                                }
                            }
                            else
                                throw new BohriumException(string.Format("Unexpected data type: {0}", typeof(T).FullName));
                        }

                        VEM.Execute(new PInvoke.bh_instruction(bh_opcode.BH_FREE, m_externalData.Pointer));
                    }
                }

                m_externalData.Dispose();
                m_externalData = null;
            }
        }

        /// <summary>
        /// Allocates the data either in Bohrium or in managed memory
        /// </summary>
        public override void Allocate()
        {
            this.EnsureSynced();
        }

        /// <summary>
        /// Gets a pointer to data
        /// </summary>
        public IntPtr Pointer
        {
            get 
            {
                EnsureSynced();

                System.Diagnostics.Debug.Assert(m_data != null || m_externalData != null);

                if (m_data != null)
                {
                    if (m_externalData != null && !m_externalData.HasHandle)
                    {
                        m_externalData.Dispose();
                        m_externalData = null;
                    }

                    if (m_externalData == null || !m_externalData.HasHandle)
                    {
                        GCHandle h = GCHandle.Alloc(m_data, GCHandleType.Pinned);
                        PInvoke.bh_array_ptr p = VEM.CreateBaseArray(m_data);
                        p.Data = h.AddrOfPinnedObject();
                        m_externalData = new ViewPtrKeeper(p, h);
                    }
                }
                else
                {
                    System.Diagnostics.Debug.Assert(m_externalData != null && m_externalData.Pointer.Data != IntPtr.Zero);
                }

                return m_externalData.Pointer.Data;
            }
        }

        /// <summary>
        /// Gets a pointer to the base array
        /// </summary>
        public PInvoke.bh_array_ptr BaseArrayPtr
        {
            get
            {
                if (m_data == null && m_externalData == null)
                {
                    m_externalData = new ViewPtrKeeper(VEM.CreateBaseArray(BH_TYPE, m_size));
                    return m_externalData.Pointer;
                }

                if (m_externalData != null)
                    return m_externalData.Pointer;

                if (m_data != null)
                {
                    GCHandle h = GCHandle.Alloc(m_data, GCHandleType.Pinned);
                    PInvoke.bh_array_ptr p = VEM.CreateBaseArray(m_data);
                    p.Data = h.AddrOfPinnedObject();
                    m_externalData = new ViewPtrKeeper(p, h);

                    return m_externalData.Pointer;
                }

                throw new Exception("An assumption failed");
            }
        }

        /// <summary>
        /// Returns a value indicating if the data can be allocated as a managed array
        /// </summary>
        public bool CanAllocateArray
        {
            get { return m_size < int.MaxValue; }
        }

		public override void DoExecute(IList<IPendingOperation> work)
		{
			var tmp = new List<IPendingOperation>();
			var continuationList = new List<IInstruction>();
			while (work.Count > 0)
			{
				var pendingOpType = typeof(PendingOperation<>).MakeGenericType(new Type[] { work[0].DataType });
				var enumType = typeof(IEnumerable<>).MakeGenericType(new Type[] { pendingOpType } );
				
				while (work.Count > 0 && (tmp.Count == 0 || work[0].TargetOperandType == tmp[0].TargetOperandType))
				{
					tmp.Add(work[0]);
					work.RemoveAt(0);
				}
				
				var typedEnum = typeof(LazyAccessorCollector).GetMethod("ConvertList", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.DeclaredOnly | System.Reflection.BindingFlags.Static, null, new Type[] { typeof(System.Collections.IEnumerable) }, null).MakeGenericMethod(new Type[] { pendingOpType }).Invoke(null, new object[] { tmp });
				if (tmp[0].TargetOperandType.GetGenericTypeDefinition() == typeof(BohriumAccessor<>))
				{
					tmp[0].TargetOperandType.GetMethod("DoExecute", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.FlattenHierarchy | System.Reflection.BindingFlags.Instance, null, new Type[] { enumType, typeof(List<IInstruction>) }, null ).Invoke(tmp[0].TargetAccessor, new object[] { typedEnum, continuationList });
				}
				else
				{
					if (continuationList.Count > 0)
					{
						ExecuteWithFailureDetection(continuationList);
						continuationList.Clear();
					}
					
					tmp[0].TargetOperandType.GetMethod("DoExecute", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.FlattenHierarchy | System.Reflection.BindingFlags.Instance, null, new Type[] { enumType }, null ).Invoke(tmp[0].TargetAccessor, new object[] { typedEnum });
				}
				
				tmp.Clear();
			}
			
			if (continuationList.Count > 0)
				ExecuteWithFailureDetection(continuationList);
		}
		
		/// <summary>
		/// Gets the opcode for the reduce operation, given the reduction operation
		/// </summary>
		/// <returns>The reduce opcode.</returns>
		/// <param name="operation">The operation to examine</param>
		private bh_opcode GetReduceOpCode(IBinaryOp<T> operation)
		{
			bh_opcode reduce_opcode;
			if (OpcodeMap.TryGetValue(operation.GetType(), out reduce_opcode))
			{
				switch(reduce_opcode)
				{
					case bh_opcode.BH_ADD:
						return bh_opcode.BH_ADD_REDUCE;
					case bh_opcode.BH_MULTIPLY:
						return bh_opcode.BH_MULTIPLY_REDUCE;
					case bh_opcode.BH_MINIMUM:
						return bh_opcode.BH_MINIMUM_REDUCE;
					case bh_opcode.BH_MAXIMUM:
						return bh_opcode.BH_MAXIMUM_REDUCE;
					case bh_opcode.BH_BITWISE_AND:
						return bh_opcode.BH_BITWISE_AND_REDUCE;
					case bh_opcode.BH_BITWISE_OR:
						return bh_opcode.BH_BITWISE_OR_REDUCE;
					case bh_opcode.BH_BITWISE_XOR:
						return bh_opcode.BH_BITWISE_XOR_REDUCE;
					case bh_opcode.BH_LOGICAL_AND:
						return bh_opcode.BH_LOGICAL_AND_REDUCE;
					case bh_opcode.BH_LOGICAL_OR:
						return bh_opcode.BH_LOGICAL_OR_REDUCE;
					case bh_opcode.BH_LOGICAL_XOR:
						return bh_opcode.BH_LOGICAL_XOR_REDUCE;
				}
			}
			
			return bh_opcode.BH_NONE;
		}
		
		/// <summary>
		/// Executes all pending operations in the list
		/// </summary>
		/// <param name="work">The list of operations to execute</param>
		/// <param name="supported">A list of supported instructions that is produced from another context</param>
		public override void DoExecute(IEnumerable<PendingOperation<T>> work)
		{
			DoExecute(work, null);
		}
		
        /// <summary>
        /// Executes all pending operations in the list
        /// </summary>
        /// <param name="work">The list of operations to execute</param>
        /// <param name="supported">A list of supported instructions that is produced from another context</param>
        private void DoExecute(IEnumerable<PendingOperation<T>> work, List<IInstruction> supported)
        {
            var unsupported = new List<PendingOperation<T>>();
            bool isContinuation = supported != null;

            if (supported == null)
                supported = new List<IInstruction>();

            foreach (var op in work)
            {
                IOp<T> ops = op.Operation;
                NdArray<T>[] operands = op.Operands;
                
                bh_opcode opcode;
                if (OpcodeMap.TryGetValue(ops.GetType(), out opcode))
                {
                    if (unsupported.Count > 0)
                    {
						Console.WriteLine("[Warning] Executing {0} unsupported operation(s)", unsupported.Count);
						foreach(var un in unsupported)
							Console.WriteLine(un.Operation);

                        base.DoExecute(unsupported);
                        unsupported.Clear();
                    }

                    bool isSupported = true;

                    if (opcode == bh_opcode.BH_USERFUNC)
                    {
						isSupported = false;
						
						if (VEM.SupportsRandom && ops is NumCIL.Generic.IRandomGeneratorOp<T>)
                        {
                            //Bohrium only supports random for plain arrays
                            if (operands[0].Shape.IsPlain && operands[0].Shape.Offset == 0 && operands[0].Shape.Elements == operands[0].DataAccessor.Length)
                            {
                                supported.Add(VEM.CreateRandomInstruction<T>(BH_TYPE, operands[0]));
                                isSupported = true;
                            }
                        }
                        else if (VEM.SupportsMatmul && ops is NumCIL.UFunc.LazyMatmulOperation<T>)
                        {
                            supported.Add(VEM.CreateMatmulInstruction<T>(BH_TYPE, operands[0], operands[1], operands[2]));
                            isSupported = true;
                        }
						else if (ops is NumCIL.UFunc.LazyReduceOperation<T>)
						{
							NumCIL.UFunc.LazyReduceOperation<T> lzop = (NumCIL.UFunc.LazyReduceOperation<T>)op.Operation;
							bh_opcode rop = GetReduceOpCode(lzop.Operation);
							if (rop != bh_opcode.BH_NONE)
							{
								supported.Add(VEM.CreateInstruction<T>(BH_TYPE, rop, operands[0], operands[1], new PInvoke.bh_constant(lzop.Axis)));
								isSupported = true;
							}
						} 
						else if (ops is NumCIL.UFunc.LazyAggregateOperation<T>)
						{
							NumCIL.UFunc.LazyAggregateOperation<T> lzop = (NumCIL.UFunc.LazyAggregateOperation<T>)op.Operation;
							bh_opcode rop = GetReduceOpCode(lzop.Operation);
							if (rop != bh_opcode.BH_NONE)
							{
								var sourceOp = operands[1];
								NumCIL.Generic.NdArray<T> targetOp;
								
								if (sourceOp.Shape.Dimensions.LongLength > 1)
								{
									do
									{
										var targetShape = new Shape.ShapeDimension[sourceOp.Shape.Dimensions.LongLength - 1];
										Array.Copy(sourceOp.Shape.Dimensions, targetShape, targetShape.LongLength);
										targetOp = new NumCIL.Generic.NdArray<T>(new Shape(targetShape));
										
										supported.Add(VEM.CreateInstruction<T>(BH_TYPE, rop, targetOp, sourceOp, new PInvoke.bh_constant(targetOp.Shape.Dimensions.LongLength)));
										sourceOp = targetOp;
										
									} while(targetOp.Shape.Dimensions.LongLength > 1);
								}
								
								supported.Add(VEM.CreateInstruction<T>(BH_TYPE, rop, operands[0], sourceOp, new PInvoke.bh_constant(0L)));
								isSupported = true;
							}
						}
												
						if (!isSupported)
                        {
                            if (supported.Count > 0)
                                ExecuteWithFailureDetection(supported);

                            unsupported.Add(op);
                        }
                    }
                    else
                    {
                        if (op is IPendingUnaryConversionOp && opcode == bh_opcode.BH_IDENTITY)
                        {
                            //As we cross execution spaces, we need to ensure that the input operand has no pending instructions
                            object unop = ((IPendingUnaryConversionOp)op).InputOperand;

                            Type inputType = unop.GetType().GetGenericArguments()[0];
                            IInstruction inst = (IInstruction)VEMConversionMethod.MakeGenericMethod(typeof(T), inputType).Invoke(VEM, new object[] { supported, opcode,  BH_TYPE, operands[0], ((IPendingUnaryConversionOp)op).InputOperand, null });

                            supported.Add(inst);
                        }
                        else if (op is IPendingBinaryConversionOp)
                        {
                            //As we cross execution spaces, we need to ensure that the input operands has no pending instructions
                            object lhsop = ((IPendingUnaryConversionOp)op).InputOperand;
                            object rhsop = ((IPendingBinaryConversionOp)op).InputOperand;

                            Type inputType = lhsop.GetType().GetGenericArguments()[0];
                            IInstruction inst = (IInstruction)VEMConversionMethod.MakeGenericMethod(typeof(T), inputType).Invoke(VEM, new object[] { supported, opcode, BH_TYPE, operands[0], lhsop, rhsop });

                            supported.Add(inst);
                        } 
                        else
                        {
                        	IInstruction inst;
                            if (operands.Length == 1)
								inst = VEM.CreateInstruction<T>(BH_TYPE, opcode, operands[0]);
                            else if (operands.Length == 2)
                                inst = VEM.CreateInstruction<T>(BH_TYPE, opcode, operands[0], operands[1]);
                            else if (operands.Length == 3)
                                inst = VEM.CreateInstruction<T>(BH_TYPE, opcode, operands[0], operands[1], operands[2]);
                            else
                                inst = VEM.CreateInstruction<T>(BH_TYPE, opcode, operands);
                                
                            if (VEM.IsValidInstruction(inst))
                            	supported.Add(inst);
                            else
                            {
								/*var conv = VEM.GetConversionSequence(inst);
								if (conv != null)
									supported.AddRange(conv);
								else*/
								{
									if (supported.Count > 0)
										ExecuteWithFailureDetection(supported);
									
									unsupported.Add(op);
								}
                            }
                        }
                    }
                }
                else
                {
                    if (supported.Count > 0)
                        ExecuteWithFailureDetection(supported);

                    unsupported.Add(op);
                }
            }

            if (supported.Count > 0 && unsupported.Count > 0)
                throw new InvalidOperationException("Unexpected result, both supported and non-supported operations");

            if (unsupported.Count > 0)
			{
				Console.WriteLine("[Warning] Executing {0} unsupported operation(s)", unsupported.Count);
				foreach(var un in unsupported)
					Console.WriteLine(un.Operation);
                base.DoExecute(unsupported);
			}

            if (supported.Count > 0 && !isContinuation)
            {
                ExecuteWithFailureDetection(supported);
            }
        }

        /// <summary>
        /// Performs GC-gen0 collection and then executes the instrucions in the list
        /// </summary>
        /// <param name="instructions">The list of instructions to execute</param>
        protected static void ExecuteWithFailureDetection(List<IInstruction> instructions)
        {
            //Yield to the GC
            GC.Collect();
			GC.WaitForPendingFinalizers();
			
			VEM.Execute(instructions);
            instructions.Clear();
            return;
        }

        /// <summary>
        /// Disposes all resources held by this instance
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        /// Disposes all resources held by this instance
        /// </summary>
        /// <param name="disposing">True if called from the </param>
        protected void Dispose(bool disposing)
        {
            if (m_externalData != null)
                m_externalData.Dispose();
            m_externalData = null;
            m_data = null;

            if (disposing)
                GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finializes this object
        /// </summary>
        ~BohriumAccessor()
        {
            Dispose(false);
        }
    }
}
