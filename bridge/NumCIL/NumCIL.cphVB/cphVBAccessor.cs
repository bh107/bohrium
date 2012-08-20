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
using NumCIL.Generic;
using System.Runtime.InteropServices;

namespace NumCIL.cphVB
{
    /// <summary>
    /// Basic factory for creating cphVB accessors
    /// </summary>
    /// <typeparam name="T">The type of data kept in the underlying array</typeparam>
    public class cphVBAccessorFactory<T> : NumCIL.Generic.IAccessorFactory<T>
    {
        /// <summary>
        /// Creates a new accessor for a data chunk of the given size
        /// </summary>
        /// <param name="size">The size of the array</param>
        /// <returns>An accessor</returns>
        public IDataAccessor<T> Create(long size) { return size == 1 ? new cphVBAccessor<T>(new T[1]) : new cphVBAccessor<T>(size); }
        /// <summary>
        /// Creates a new accessor for a preallocated array
        /// </summary>
        /// <param name="data">The data to wrap</param>
        /// <returns>An accessor</returns>
        public IDataAccessor<T> Create(T[] data) { return new cphVBAccessor<T>(data); }
    }
    
    /// <summary>
    /// Code to map from NumCIL operations to cphVB operations
    /// </summary>
    public class OpCodeMapper
    {
        /// <summary>
        /// Lookup table with mapping from NumCIL operation name to cphVB opcode
        /// </summary>
        private static Dictionary<cphvb_opcode, string> _opcode_func_name;

        /// <summary>
        /// Static initializer, builds mapping table between the cphVB opcodes.
        /// and the corresponding names of the operations in NumCIL
        /// </summary>
        static OpCodeMapper()
        {
            _opcode_func_name = new Dictionary<cphvb_opcode, string>();

            _opcode_func_name.Add(cphvb_opcode.CPHVB_ADD, "Add");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_SUBTRACT, "Sub");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_MULTIPLY, "Mul");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_DIVIDE, "Div");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_MOD, "Mod");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_MAXIMUM, "Max");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_MINIMUM, "Min");

            //These two are not found in cphVB, but are emulated with ADD and SUB
            //_opcode_func_name.Add(cphvb_opcode.CPHVB_INCREMENT, "Inc");
            //_opcode_func_name.Add(cphvb_opcode.CPHVB_DECREMENT, "Dec");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_FLOOR, "Floor");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_CEIL, "Ceiling");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_RINT, "Round");

            _opcode_func_name.Add(cphvb_opcode.CPHVB_ABSOLUTE, "Abs");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_SQRT, "Sqrt");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_EXP, "Exp");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_LOG, "Log");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_LOG10, "Log10");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_POWER, "Pow");

            _opcode_func_name.Add(cphvb_opcode.CPHVB_COS, "Cos");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_SIN, "Sin");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_TAN, "Tan");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_ARCCOS, "Acos");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_ARCSIN, "Asin");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_ARCTAN, "Atan");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_COSH, "Cosh");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_SINH, "Sinh");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_TANH, "Tanh");

            _opcode_func_name.Add(cphvb_opcode.CPHVB_LOGICAL_NOT, "Not");
            _opcode_func_name.Add(cphvb_opcode.CPHVB_INVERT, "Invert");
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
        /// <returns>A mapping between the type used for this executor and the cphVB opcodes</returns>
        public static Dictionary<Type, cphvb_opcode> CreateOpCodeMap<T>()
        {
            Dictionary<Type, cphvb_opcode> res = new Dictionary<Type, cphvb_opcode>();

            Type basic = GetBasicClass<T>();
            Dictionary<cphvb_opcode, string> opcodenames = new Dictionary<cphvb_opcode, string>(_opcode_func_name);

            if (typeof(T) == typeof(bool))
            {
                opcodenames.Add(cphvb_opcode.CPHVB_LOGICAL_AND, "And");
                opcodenames.Add(cphvb_opcode.CPHVB_LOGICAL_OR, "Or");
                opcodenames.Add(cphvb_opcode.CPHVB_LOGICAL_XOR, "Xor");
            }
            else
            {
                opcodenames.Add(cphvb_opcode.CPHVB_BITWISE_AND, "And");
                opcodenames.Add(cphvb_opcode.CPHVB_BITWISE_OR, "Or");
                opcodenames.Add(cphvb_opcode.CPHVB_BITWISE_XOR, "Xor");
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

            res[typeof(NumCIL.CopyOp<T>)] = cphvb_opcode.CPHVB_IDENTITY;
            res[typeof(NumCIL.GenerateOp<T>)] = cphvb_opcode.CPHVB_IDENTITY;
            if (VEM.Instance.SupportsRandom)
			{
                res[typeof(NumCIL.Generic.IRandomGeneratorOp<T>)] = cphvb_opcode.CPHVB_USERFUNC;
				try { res[basic.Assembly.GetType("NumCIL.Generic.RandomGeneratorOp" + typeof(T).Name)] = cphvb_opcode.CPHVB_USERFUNC; }
				catch {}
			}
            if (VEM.Instance.SupportsReduce)
                res[typeof(NumCIL.UFunc.LazyReduceOperation<T>)] = cphvb_opcode.CPHVB_USERFUNC;
            if (VEM.Instance.SupportsMatmul)
                res[typeof(NumCIL.UFunc.LazyMatmulOperation<T>)] = cphvb_opcode.CPHVB_USERFUNC;


            if (typeof(T) == typeof(NumCIL.Complex64.DataType))
            {
                res[typeof(NumCIL.Complex64.ToComplex)] = cphvb_opcode.CPHVB_IDENTITY;
            }
            else if (typeof(T) == typeof(System.Numerics.Complex))
            {
                res[typeof(NumCIL.Complex128.ToComplex)] = cphvb_opcode.CPHVB_IDENTITY;
            }
            else
            {
                foreach (var e in new string[] {"Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64", "Float", "Double"})
                {
                    try 
                    {
                        Type t = basic.Assembly.GetType(basic.Namespace + ".To" + e);
                        if (t != null)
                            res[t] = cphvb_opcode.CPHVB_IDENTITY; 
                    }
                    catch { }
                }
            }

            if (typeof(T) == typeof(bool))
            {
                Dictionary<cphvb_opcode, string> logicalnames = new Dictionary<cphvb_opcode, string>();
                logicalnames.Add(cphvb_opcode.CPHVB_EQUAL, "Equal");
                logicalnames.Add(cphvb_opcode.CPHVB_NOT_EQUAL, "NotEqual");
                logicalnames.Add(cphvb_opcode.CPHVB_GREATER, "GreaterThan");
                logicalnames.Add(cphvb_opcode.CPHVB_LESS, "LessThan");
                logicalnames.Add(cphvb_opcode.CPHVB_GREATER_EQUAL, "GreaterThanOrEqual");
                logicalnames.Add(cphvb_opcode.CPHVB_LESS_EQUAL, "LessThanOrEqual");

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
                        res[t] = cphvb_opcode.CPHVB_IDENTITY;
                }
                catch
                {
                }
            }


            return res;
        }
    }

    /// <summary>
    /// Basic accessor for a cphVB array
    /// </summary>
    /// <typeparam name="T">The type of data kept in the underlying array</typeparam>
    public class cphVBAccessor<T> : NumCIL.Generic.LazyAccessor<T>, IDisposable, IUnmanagedDataAccessor<T>
    {
        /// <summary>
        /// Lock that prevents multithreaded access to the cphVB data
        /// </summary>
        private readonly object m_lock = new object();

        /// <summary>
        /// Instance of the VEM that is used
        /// </summary>
        protected static VEM VEM = NumCIL.cphVB.VEM.Instance;

        /// <summary>
        /// The maximum number of instructions to queue
        /// </summary>
        protected static readonly long HIGH_WATER_MARK = 4000;

        /// <summary>
        /// Local copy of the type, to avoid lookups in the VEM dictionary
        /// </summary>
        protected static readonly PInvoke.cphvb_type CPHVB_TYPE = VEM.MapType(typeof(T));

        /// <summary>
        /// The size of the data element in native code
        /// </summary>
        protected static readonly int NATIVE_ELEMENT_SIZE = Marshal.SizeOf(typeof(T));

        /// <summary>
        /// A lookup table that maps NumCIL operation types to cphVB opcodes
        /// </summary>
        protected static Dictionary<Type, cphvb_opcode> OpcodeMap = OpCodeMapper.CreateOpCodeMap<T>();

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
        public cphVBAccessor(long size) : base(size) { }

        /// <summary>
        /// Constructs a new data accessor for a pre-allocated block of storage
        /// </summary>
        /// <param name="data"></param>
        public cphVBAccessor(T[] data) : base(data) { }

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
                VEM.Execute(new PInvoke.cphvb_instruction(cphvb_opcode.CPHVB_SYNC, m_externalData.Pointer));
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
                        throw new cphVBException(string.Format("Unexpected data type: {0}", typeof(T).FullName));
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
                                throw new cphVBException(string.Format("Unexpected data type: {0}", typeof(T).FullName));
                        }

                        VEM.Execute(new PInvoke.cphvb_instruction(cphvb_opcode.CPHVB_FREE, m_externalData.Pointer));
                    }
                }

                m_externalData.Dispose();
                m_externalData = null;
            }
        }

        /// <summary>
        /// Allocates the data either in cphvb or in managed memory
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
                        PInvoke.cphvb_array_ptr p = VEM.CreateBaseArray(m_data);
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
        public PInvoke.cphvb_array_ptr BaseArrayPtr
        {
            get
            {
                if (m_data == null && m_externalData == null)
                {
                    m_externalData = new ViewPtrKeeper(VEM.CreateBaseArray(CPHVB_TYPE, m_size));
                    return m_externalData.Pointer;
                }

                if (m_externalData != null)
                    return m_externalData.Pointer;

                if (m_data != null)
                {
                    GCHandle h = GCHandle.Alloc(m_data, GCHandleType.Pinned);
                    PInvoke.cphvb_array_ptr p = VEM.CreateBaseArray(m_data);
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

        /// <summary>
        /// Continues exectuion started on another type
        /// </summary>
        /// <param name="i">The execution so far was started on</param>
        internal void ContinueExecution(List<IInstruction> i)
        {
            lock (Lock)
            {
                var lst = UnrollWorkList(this);
                PendingOperations.Clear();
                ExecuteOperations(lst, i);
            }
        }

        /// <summary>
        /// Executes all pending operations in the list
        /// </summary>
        /// <param name="work">The list of operations to execute</param>
        public override void ExecuteOperations(IEnumerable<PendingOperation<T>> work)
        {
            ExecuteOperations(work, null);
        }

        /// <summary>
        /// Executes all pending operations in the list
        /// </summary>
        /// <param name="work">The list of operations to execute</param>
        /// <param name="supported">A list of supported instructions that is produced from another context</param>
        private void ExecuteOperations(IEnumerable<PendingOperation<T>> work, List<IInstruction> supported)
        {
            List<PendingOperation<T>> unsupported = new List<PendingOperation<T>>();
            bool isContinuation = supported != null;

            if (supported == null)
                supported = new List<IInstruction>();

            foreach (var op in work)
            {
                IOp<T> ops = op.Operation;
                NdArray<T>[] operands = op.Operands;
                
                cphvb_opcode opcode;
                if (OpcodeMap.TryGetValue(ops.GetType(), out opcode))
                {
                    if (unsupported.Count > 0)
                    {
                        base.ExecuteOperations(unsupported);
                        unsupported.Clear();
                    }

                    bool isSupported = true;

                    if (opcode == cphvb_opcode.CPHVB_USERFUNC)
                    {
                        if (VEM.SupportsRandom && ops is NumCIL.Generic.IRandomGeneratorOp<T>)
                        {
                            //cphVB only supports random for plain arrays
                            if (operands[0].Shape.IsPlain && operands[0].Shape.Offset == 0 && operands[0].Shape.Elements == operands[0].DataAccessor.Length)
                            {
                                supported.Add(VEM.CreateRandomInstruction<T>(CPHVB_TYPE, operands[0]));
                                isSupported = true;
                            }
                        }
                        else if (VEM.SupportsReduce && ops is NumCIL.UFunc.LazyReduceOperation<T>)
                        {
                            NumCIL.UFunc.LazyReduceOperation<T> lzop = (NumCIL.UFunc.LazyReduceOperation<T>)op.Operation;
                            cphvb_opcode rop;
                            if (OpcodeMap.TryGetValue(lzop.Operation.GetType(), out rop))
                            {
                                supported.Add(VEM.CreateReduceInstruction<T>(CPHVB_TYPE, rop, lzop.Axis, operands[0], operands[1]));
                                isSupported = true;
                            }
                        }
                        else if (VEM.SupportsMatmul && ops is NumCIL.UFunc.LazyMatmulOperation<T>)
                        {
                            supported.Add(VEM.CreateMatmulInstruction<T>(CPHVB_TYPE, operands[0], operands[1], operands[2]));

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
                        if (op is IPendingUnaryConversionOp && opcode == cphvb_opcode.CPHVB_IDENTITY)
                        {
                            //As we cross execution spaces, we need to ensure that the input operand has no pending instructions
                            object unop = ((IPendingUnaryConversionOp)op).InputOperand;

                            Type inputType = unop.GetType().GetGenericArguments()[0];
                            IInstruction inst = (IInstruction)VEMConversionMethod.MakeGenericMethod(typeof(T), inputType).Invoke(VEM, new object[] { supported, opcode,  CPHVB_TYPE, operands[0], ((IPendingUnaryConversionOp)op).InputOperand, null });

                            supported.Add(inst);
                        }
                        else if (op is IPendingBinaryConversionOp)
                        {
                            //As we cross execution spaces, we need to ensure that the input operands has no pending instructions
                            object lhsop = ((IPendingUnaryConversionOp)op).InputOperand;
                            object rhsop = ((IPendingBinaryConversionOp)op).InputOperand;

                            Type inputType = lhsop.GetType().GetGenericArguments()[0];
                            IInstruction inst = (IInstruction)VEMConversionMethod.MakeGenericMethod(typeof(T), inputType).Invoke(VEM, new object[] { supported, opcode, CPHVB_TYPE, operands[0], lhsop, rhsop });

                            supported.Add(inst);
                        } 
                        else
                        {
                            if (operands.Length == 1)
                                supported.Add(VEM.CreateInstruction<T>(CPHVB_TYPE, opcode, operands[0]));
                            else if (operands.Length == 2)
                                supported.Add(VEM.CreateInstruction<T>(CPHVB_TYPE, opcode, operands[0], operands[1]));
                            else if (operands.Length == 3)
                                supported.Add(VEM.CreateInstruction<T>(CPHVB_TYPE, opcode, operands[0], operands[1], operands[2]));
                            else
                                supported.Add(VEM.CreateInstruction<T>(CPHVB_TYPE, opcode, operands));
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
                base.ExecuteOperations(unsupported);

            if (supported.Count > 0 && !isContinuation)
            {
                ExecuteWithFailureDetection(supported);
            }
        }

        /// <summary>
        /// Performs GC-gen0 collection and then executes the instrucions in the list
        /// </summary>
        /// <param name="instructions">The list of instructions to execute</param>
        protected void ExecuteWithFailureDetection(List<IInstruction> instructions)
        {
            //Reclaim everything in gen 0
            GC.Collect(0);

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
        ~cphVBAccessor()
        {
            Dispose(false);
        }
    }
}
