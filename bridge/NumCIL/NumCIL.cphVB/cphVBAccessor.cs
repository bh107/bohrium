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
        public IDataAccessor<T> Create(long size) { return new cphVBAccessor<T>(size); }
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
        private static Dictionary<PInvoke.cphvb_opcode, string> _opcode_func_name;

        /// <summary>
        /// Static initializer, builds mapping table between the cphVB opcodes.
        /// and the corresponding names of the operations in NumCIL
        /// </summary>
        static OpCodeMapper()
        {
            _opcode_func_name = new Dictionary<PInvoke.cphvb_opcode, string>();

            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_ADD, "Add");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_SUBTRACT, "Sub");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_MULTIPLY, "Mul");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_DIVIDE, "Div");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_MOD, "Mod");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_MAXIMUM, "Max");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_MINIMUM, "Min");

            //These two are not found in cphVB, but are emulated with ADD and SUB
            //_opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_INCREMENT, "Inc");
            //_opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_DECREMENT, "Dec");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_FLOOR, "Floor");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_CEIL, "Ceiling");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_RINT, "Round");

            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_ABSOLUTE, "Abs");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_SQRT, "Sqrt");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_EXP, "Exp");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_NEGATIVE, "Negate");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_LOG, "Log");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_LOG10, "Log10");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_POWER, "Pow");
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
            else
                throw new Exception("Unexpected type: " + (typeof(T)).FullName);            
        }

        /// <summary>
        /// Helper function to get the opcode mapping table for the current type
        /// </summary>
        /// <returns>A mapping between the type used for this executor and the cphVB opcodes</returns>
        public static Dictionary<Type, PInvoke.cphvb_opcode> CreateOpCodeMap<T>()
        {
            Dictionary<Type, PInvoke.cphvb_opcode> res = new Dictionary<Type, PInvoke.cphvb_opcode>();

            Type basic = GetBasicClass<T>();

            foreach (var e in _opcode_func_name)
            {
                try { res[basic.Assembly.GetType(basic.Namespace + "." + e.Value)] = e.Key; }
                catch { }
            }

            res[typeof(NumCIL.CopyOp<T>)] = PInvoke.cphvb_opcode.CPHVB_IDENTITY;
            res[typeof(NumCIL.GenerateOp<T>)] = PInvoke.cphvb_opcode.CPHVB_IDENTITY;
            if (VEM.Instance.SupportsRandom)
                res[typeof(NumCIL.Generic.RandomGeneratorOp<T>)] = PInvoke.cphvb_opcode.CPHVB_USERFUNC;
            if (VEM.Instance.SupportsReduce)
                res[typeof(NumCIL.UFunc.LazyReduceOperation<T>)] = PInvoke.cphvb_opcode.CPHVB_USERFUNC;
            if (VEM.Instance.SupportsMatmul)
                res[typeof(NumCIL.UFunc.LazyMatmulOperation<T>)] = PInvoke.cphvb_opcode.CPHVB_USERFUNC;
            return res;
        }
    }

    public class PendingOpCounter<T> : PendingOperation<T>, IDisposable
    {
        private static long _pendingOpCount = 0;
        public static long PendingOpCount { get { return _pendingOpCount; } }
        private bool m_isDisposed = false;

        public PendingOpCounter(IOp<T> operation, params NdArray<T>[] operands)
            : base(operation, operands)
        {
            System.Threading.Interlocked.Increment(ref _pendingOpCount);
        }

        protected void Dispose(bool disposing)
        {
            if (!m_isDisposed)
            {
                System.Threading.Interlocked.Decrement(ref _pendingOpCount);
                m_isDisposed = true;

                if (disposing)
                    GC.SuppressFinalize(this);
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }

        ~PendingOpCounter()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// Basic accessor for a cphVB array
    /// </summary>
    /// <typeparam name="T">The type of data kept in the underlying array</typeparam>
    public class cphVBAccessor<T> : NumCIL.Generic.LazyAccessor<T>, IDisposable, IUnmanagedDataAccessor<T>
    {
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
        protected static Dictionary<Type, PInvoke.cphvb_opcode> OpcodeMap = OpCodeMapper.CreateOpCodeMap<T>();

        /// <summary>
        /// Gets the type for the Increment operation
        /// </summary>
        protected static readonly Type IncrementOp = OpCodeMapper.GetOp<T>("Inc");

        /// <summary>
        /// Gets the type for the Decrement operation
        /// </summary>
        protected static readonly Type DecrementOp = OpCodeMapper.GetOp<T>("Dec");

        /// <summary>
        /// Gets the type for the Add operation
        /// </summary>
        protected static readonly Type AddOp = OpCodeMapper.GetOp<T>("Add");

        /// <summary>
        /// Gets the type for the Sub operation
        /// </summary>
        protected static readonly Type SubOp = OpCodeMapper.GetOp<T>("Sub");

        /// <summary>
        /// The constant 1
        /// </summary>
        protected static readonly T ONE = (T)Convert.ChangeType(1, typeof(T)); 

        /// <summary>
        /// Constructs a new data accessor for the given size
        /// </summary>
        /// <param name="size">The size of the data</param>
        public cphVBAccessor(long size) : base(size) { }

        /// <summary>
        /// Constructs a new data accessor for a pre-allocated block of storage
        /// </summary>
        /// <param name="data"></param>
        public cphVBAccessor(T[] data) : base(data) { m_ownsData = true; }

        /// <summary>
        /// A pointer to the base-array view structure
        /// </summary>
        protected PInvoke.cphvb_array_ptr m_externalData = PInvoke.cphvb_array_ptr.Null;

        /// <summary>
        /// A value indicating if NumCIL owns the data, false means that cphVB owns the data
        /// </summary>
        protected bool m_ownsData = false;

        /// <summary>
        /// A value indicating if a CPHVB_SYNC command has been sent
        /// </summary>
        protected bool m_isSynced = false;

        /// <summary>
        /// A value indicating if a CPHVB_DISCARD command has been sent
        /// </summary>
        protected bool m_isDiscarded = false;

        /// <summary>
        /// A pointer to internally allocated data which is pinned
        /// </summary>
        protected GCHandle m_handle;

        /// <summary>
        /// Ensures the underlying data block is flushed and updated
        /// </summary>
        public override void Allocate()
        {
            Flush();

            if (IsAllocated)
            {
                cphVB_SyncAndDiscard();
            }
            else
            {
                if (m_data == null && m_externalData == PInvoke.cphvb_array_ptr.Null)
                {
                    //Data is not yet allocated, convert to external storage
                    m_externalData = VEM.CreateArray(CPHVB_TYPE, m_size);
                    m_ownsData = false;
                }

                if (!m_ownsData && m_externalData.Data == IntPtr.Zero)
                {
                    PInvoke.cphvb_error e = PInvoke.cphvb_data_malloc(m_externalData);
                    if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                        throw new cphVBException(e);
                }
            }

        }

        /// <summary>
        /// Returns the internal data as an array
        /// </summary>
        /// <returns>The data as a managed array</returns>
        public override T[] AsArray()
        {
            MakeDataManaged();
            return base.AsArray();
        }

        /// <summary>
        /// Gets a value describing if the data is allocated or not
        /// </summary>
        public override bool IsAllocated
        {
            get
            {
                return m_data != null || (m_externalData != PInvoke.cphvb_array_ptr.Null && m_externalData.Data != IntPtr.Zero);
            }
        }

        /// <summary>
        /// Gets or sets a value in the array
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public override T this[long index]
        {
            get
            {
                Allocate();

                if (m_data == null)
                {
                    IntPtr ptr = new IntPtr(Pointer.ToInt64() + (index * NATIVE_ELEMENT_SIZE));
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
                    else
                        throw new cphVBException(string.Format("Unexpected data type: {0}", typeof(T).FullName));
                }
                else
                    return base[index];
            }
            set
            {
                Allocate();

                if (m_data == null)
                {
                    IntPtr ptr = new IntPtr(Pointer.ToInt64() + (index * NATIVE_ELEMENT_SIZE));
                    if (typeof(T) == typeof(float))
                    {
                        T[] tmp = new T[] { value };
                        if (!NumCIL.UnsafeAPI.CopyToIntPtr(tmp, ptr, 1))
                            Marshal.Copy((float[])(object)tmp, 0, ptr, 1);
                    }
                    else if (typeof(T) == typeof(double))
                    {
                        T[] tmp = new T[] { value };
                        if (!NumCIL.UnsafeAPI.CopyToIntPtr(tmp, ptr, 1))
                            Marshal.Copy((double[])(object)tmp, 0, ptr, 1);
                    }
                    else if (typeof(T) == typeof(sbyte))
                        Marshal.WriteByte(ptr, (byte)(object)value);
                    else if (typeof(T) == typeof(short))
                        Marshal.WriteInt16(ptr, (short)(object)value);
                    else if (typeof(T) == typeof(int))
                        Marshal.WriteInt32(ptr, (int)(object)value);
                    else if (typeof(T) == typeof(long))
                        Marshal.WriteInt64(ptr, (long)(object)value);
                    else if (typeof(T) == typeof(byte))
                        Marshal.WriteByte(ptr, (byte)(object)value);
                    else if (typeof(T) == typeof(ushort))
                        Marshal.WriteInt16(ptr, (short)(object)value);
                    else if (typeof(T) == typeof(uint))
                        Marshal.WriteInt32(ptr, (int)(object)value);
                    else if (typeof(T) == typeof(ulong))
                        Marshal.WriteInt64(ptr, (long)(object)value);
                    else
                        throw new cphVBException(string.Format("Unexpected data type: {0}", typeof(T).FullName));
                }
                else
                    base[index] = value;
            }
        }

        /// <summary>
        /// Gets a pointer to data, this will allocate data
        /// </summary>
        public IntPtr Pointer
        {
            get 
            {
                Allocate();
                cphVB_SyncAndDiscard();

                Pin();
                return m_externalData.Data; 
            }
        }

        /// <summary>
        /// Gets a value describing if the data can be allocated as a managed array
        /// </summary>
        public bool CanAllocateArray
        {
            get { return (NATIVE_ELEMENT_SIZE * m_size) < int.MaxValue; }
        }

        /// <summary>
        /// Sends a CPHVB_SYNC command to the VEM
        /// </summary>
        protected void cphVB_Sync()
        {
            if (!m_isSynced && m_externalData != PInvoke.cphvb_array_ptr.Null && m_externalData.Data != IntPtr.Zero)
            {
                VEM.Execute(new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_SYNC, m_externalData));
                m_isSynced = true;
            }
        }

        /// <summary>
        /// Sends a CPHVB_DISCARD command to the VEM
        /// </summary>
        protected void cphVB_Discard()
        {
            if (!m_isDiscarded && m_externalData != PInvoke.cphvb_array_ptr.Null && m_externalData.Data != IntPtr.Zero)
            {
                VEM.Execute(new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_DISCARD, m_externalData));
                m_isDiscarded = true;
            }
        }

        /// <summary>
        /// Sends a CPHVB_SYNC command to the VEM
        /// </summary>
        protected void cphVB_SyncAndDiscard()
        {
            if (m_externalData != PInvoke.cphvb_array_ptr.Null && m_externalData.Data != IntPtr.Zero)
            {
                if (!m_isSynced && !m_isDiscarded)
                {
                    VEM.Execute(
                        new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_SYNC, m_externalData),
                        new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_DISCARD, m_externalData)
                    );
                    m_isSynced = true;
                    m_isDiscarded = true;
                }
                else if (!m_isSynced)
                    cphVB_Sync();
                else if (!m_isDiscarded)
                    cphVB_Discard();


            }
        }


        /// <summary>
        /// Register a pending operation on the underlying array
        /// </summary>
        /// <param name="operation">The operation performed</param>
        /// <param name="operands">The operands involved, operand 0 is the target</param>
        public override void AddOperation(IOp<T> operation, params NdArray<T>[] operands)
        {
            lock (Lock)
                PendingOperations.Add(new PendingOpCounter<T>(operation, operands));

            if (PendingOpCounter<T>.PendingOpCount > HIGH_WATER_MARK)
            {
                this.Flush();
            }
        }

        /// <summary>
        /// Pins the allocated data and returns the pinned pointer
        /// </summary>
        /// <returns>A pinned pointer</returns>
        internal virtual PInvoke.cphvb_array_ptr Pin()
        {
            Allocate();
            
            if (m_externalData == PInvoke.cphvb_array_ptr.Null)
            {
                //Internally allocated data, we need to pin it
                if (!m_handle.IsAllocated)
                    m_handle = GCHandle.Alloc(m_data, GCHandleType.Pinned);

                m_externalData = VEM.CreateArray(CPHVB_TYPE, m_size);
                m_externalData.Data = m_handle.AddrOfPinnedObject();
                m_ownsData = true;

                m_isDiscarded = false;
                m_isSynced = false;
            }
            else if (m_ownsData && m_externalData.Data == IntPtr.Zero)
            {
                //Internally allocated data, we need to pin it
                if (!m_handle.IsAllocated)
                    m_handle = GCHandle.Alloc(m_data, GCHandleType.Pinned);
                m_externalData.Data = m_handle.AddrOfPinnedObject();

                m_isDiscarded = false;
                m_isSynced = false;
            }

            return m_externalData;
        }

        /// <summary>
        /// Unpins allocated data
        /// </summary>
        protected void Unpin()
        {
            cphVB_Sync();

            if (m_handle.IsAllocated)
            {
                m_handle.Free();
                m_externalData.Data = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Makes the data managed
        /// </summary>
        protected void MakeDataManaged()
        {
            if (m_size < 0)
                throw new ObjectDisposedException(this.GetType().FullName);

            if (m_ownsData && m_externalData != PInvoke.cphvb_array_ptr.Null)
            {
                this.Unpin();
                m_externalData.Data = IntPtr.Zero;
                return;
            }

            //Allocate data internally, and flush instructions as required
            base.Allocate();
            T[] data = base.m_data;


            //If data is allocated in cphVB, we need to flush it and de-allocate it
            if (m_externalData != PInvoke.cphvb_array_ptr.Null && !m_ownsData)
            {
                cphVB_Sync();

                IntPtr actualData = m_externalData.Data;
                if (actualData == IntPtr.Zero)
                {
                    //The array is "empty" which will be zeroes in NumCIL
                }
                else
                {
                    //Then copy the data into the local buffer
                    if (!NumCIL.UnsafeAPI.CopyFromIntPtr<T>(actualData, data))
                    {
                        if (m_size > int.MaxValue)
                            throw new OverflowException();

                        if (typeof(T) == typeof(float))
                            Marshal.Copy(actualData, (float[])(object)data, 0, (int)m_size);
                        else if (typeof(T) == typeof(double))
                            Marshal.Copy(actualData, (double[])(object)data, 0, (int)m_size);
                        else if (typeof(T) == typeof(sbyte))
                        {
                            sbyte[] xref = (sbyte[])(object)data;
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
                            Marshal.Copy(actualData, (short[])(object)data, 0, (int)m_size);
                        else if (typeof(T) == typeof(int))
                            Marshal.Copy(actualData, (int[])(object)data, 0, (int)m_size);
                        else if (typeof(T) == typeof(long))
                            Marshal.Copy(actualData, (long[])(object)data, 0, (int)m_size);
                        else if (typeof(T) == typeof(byte))
                            Marshal.Copy(actualData, (byte[])(object)data, 0, (int)m_size);
                        else if (typeof(T) == typeof(ushort))
                        {
                            ushort[] xref = (ushort[])(object)data;
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
                            uint[] xref = (uint[])(object)data;
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
                            ulong[] xref = (ulong[])(object)data;
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
                        else
                            throw new cphVBException(string.Format("Unexpected data type: {0}", typeof(T).FullName));
                    }
                }

                //Release the unmanaged copy
                cphVB_Discard();
                PInvoke.cphvb_data_free(m_externalData);
                m_externalData.Data = IntPtr.Zero;
                m_ownsData = true;
            }
        }


        /// <summary>
        /// Executes all pending operations in the list
        /// </summary>
        /// <param name="work">The list of operations to execute</param>
        public override void ExecuteOperations(IEnumerable<PendingOperation<T>> work)
        {
            List<PendingOperation<T>> unsupported = new List<PendingOperation<T>>();
            List<IInstruction> supported = new List<IInstruction>();
            //This list keeps all scalars so the GC does not collect them
            // before the instructions are executed, remove list once
            // constants are supported in score/mcore

            List<NdArray<T>> scalars = new List<NdArray<T>>();
            long i = 0;

            foreach (var op in work)
            {
                Type t;
                bool isScalar;
                IOp<T> ops = op.Operation;
                NdArray<T>[] operands = op.Operands;
                i++;

                if (ops is IScalarAccess<T>)
                {
                    t = ((IScalarAccess<T>)ops).Operation.GetType();
                    isScalar = true;
                }
                else
                {
                    t = ops.GetType();
                    //We mimic the Increment and Decrement with Add(1) and Add(-1) respectively
                    if (t == IncrementOp)
                    {
                        ops = new NumCIL.ScalarOp<T, IBinaryOp<T>>(ONE, (IBinaryOp<T>)Activator.CreateInstance(AddOp));
                        t = AddOp;
                        isScalar = true;
                    }
                    else if (t == DecrementOp)
                    {
                        ops = new NumCIL.ScalarOp<T, IBinaryOp<T>>(ONE, (IBinaryOp<T>)Activator.CreateInstance(SubOp));
                        t = SubOp;
                        isScalar = true;
                    }
                    else
                        isScalar = false;
                }

                PInvoke.cphvb_opcode opcode;
                if (OpcodeMap.TryGetValue(t, out opcode))
                {
                    if (unsupported.Count > 0)
                    {
                        base.ExecuteOperations(unsupported);
                        unsupported.Clear();
                    }

                    if (isScalar)
                    {
                        IScalarAccess<T> sa = (IScalarAccess<T>)ops;

                        //Change to true once score supports constants
                        if (false)
                        {
                            if (sa.Operation is IBinaryOp<T>)
                                supported.Add(VEM.CreateInstruction<T>(CPHVB_TYPE, opcode, operands[0], operands[1], new PInvoke.cphvb_constant(CPHVB_TYPE, sa.Value)));
                            else
                                supported.Add(VEM.CreateInstruction<T>(CPHVB_TYPE, opcode, operands[0], new PInvoke.cphvb_constant(CPHVB_TYPE, sa.Value)));
                        }
                        else
                        {
                            //Since we have no constant support, we mimic the constant with a 1D array
                            var scalarAcc = new cphVBAccessor<T>(1);
                            scalarAcc[0] = sa.Value;
                            NdArray<T> scalarOp;

                            if (sa.Operation is IBinaryOp<T>)
                            {
                                Shape bShape = Shape.ToBroadcastShapes(operands[1].Shape, new Shape(1)).Item2;
                                scalarOp = new NdArray<T>(scalarAcc, bShape);

                                supported.Add(VEM.CreateInstruction<T>(CPHVB_TYPE, opcode, operands[0], operands[1], scalarOp));
                            }
                            else
                            {
                                Shape bShape = Shape.ToBroadcastShapes(operands[0].Shape, new Shape(1)).Item2;
                                scalarOp = new NdArray<T>(scalarAcc, bShape);
                                supported.Add(VEM.CreateInstruction<T>(CPHVB_TYPE, opcode, operands[0], scalarOp));
                            }

                            scalars.Add(scalarOp);
                        }
                    }
                    else
                    {
                        bool isSupported = true;

                        if (opcode == PInvoke.cphvb_opcode.CPHVB_USERFUNC)
                        {
                            if (VEM.SupportsRandom && ops is NumCIL.Generic.RandomGeneratorOp<T>)
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
                                PInvoke.cphvb_opcode rop;
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
                                    ExecuteWithFailureDetection(supported, work, i - supported.Count - 1);

                                unsupported.Add(op);
                            }
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
                        ExecuteWithFailureDetection(supported, work, i - supported.Count - 1);

                    unsupported.Add(op);
                }
            }

            if (supported.Count > 0 && unsupported.Count > 0)
                throw new InvalidOperationException("Unexpected result, both supported and non-supported operations");

            if (unsupported.Count > 0)
                base.ExecuteOperations(unsupported);

            if (supported.Count > 0)
                ExecuteWithFailureDetection(supported, work, i - supported.Count);

            //TODO: Do we want to do it now, or just let the GC figure it out?
            foreach (var op in work)
                if (op is IDisposable)
                    ((IDisposable)op).Dispose();

            //Touch the data to prevent the scalars from being GC'ed
            //Remove once constants are supported
            foreach (var op in scalars)
                op.Name = null;
        }

        protected void ExecuteWithFailureDetection(List<IInstruction> instructions, IEnumerable<PendingOperation<T>> work, long instructionIndex)
        {
            //Reclaim everything in gen 0
            GC.Collect(0);

            List<PendingOperation<T>> worklist = null;
            while (instructions.Count > 0)
            {
                try
                {
                    VEM.Execute(instructions);
                    instructions.Clear();
                    return;
                }
                catch (cphVBNotSupportedInstruction cex)
                {
                    //If we get here, some arrays may be deallocated by the GC, 
                    // we need to postpone the cleanups until all instructions in the current list are completed
                    VEM.PreventCleanup = true;
                    //Remove any instructions that are reported as not being supported
                    foreach (var n in (from x in OpcodeMap where x.Value == cex.OpCode select x.Key).ToArray())
                        if (OpcodeMap.Remove(n))
                            Console.WriteLine(string.Format("Instruction was not supported by the VE: {0} -> {1}", n.FullName, cex.OpCode));

                    //We need to work on this more than once, so we convert it to a list right away
                    if (worklist == null)
                        worklist = work.Skip((int)(instructionIndex)).ToList();

                    //Extract the target instruction for inspection
                    PendingOperation<T> pt = worklist[(int)cex.InstructionNo];

                    //TODO: If the operation in question is found multiple times, 
                    // we could avoid the costly exception for each invocation

                    //Remove executed operations from each operand to prevent double flushing
                    foreach (var op in pt.Operands)
                    {
                        if (op.DataAccessor is LazyAccessor<T>)
                        {
                            ILazyAccessor<T> lzt = (ILazyAccessor<T>)op.DataAccessor;
                            if (lzt.PendingOperations.Count > 0)
                            {
                                if (lzt.PendingOperations[lzt.PendingOperations.Count - 1].Clock < pt.Clock)
                                {
                                    lzt.PendignOperationOffset += lzt.PendingOperations.Count;
                                    lzt.PendingOperations.Clear();

                                }
                                else
                                {
                                    while (lzt.PendingOperations.Count > 0 && lzt.PendingOperations[0].Clock < pt.Clock)
                                    {
                                        lzt.PendignOperationOffset++;
                                        lzt.PendingOperations.RemoveAt(0);
                                    }
                                }
                            }
                        }
                    }

                    //The unsupported instruction is now executed by the CIL
                    base.ExecuteOperations(new PendingOperation<T>[] { pt });

                    //Then remove all the instructions that have been executed
                    instructions.RemoveRange(0, (int)cex.InstructionNo + 1);
                    worklist.RemoveRange(0, (int)cex.InstructionNo + 1);
                    instructionIndex += cex.InstructionNo + 1;

                    //Make sure we manually set the data pointer to all operands that refer to the
                    // NdArray that was the target of the instruction that was excuted locally
                    NdArray<T> target = pt.Operands[0];
                    if (instructions.Count > 0)
                    {
                        int i = 0;
                        foreach (PendingOperation<T> pop in worklist)
                        {
                            if (pop.Operands.Any(x => x.DataAccessor == target.DataAccessor))
                            {
                                instructions[i] = VEM.ReCreateInstruction<T>(CPHVB_TYPE, instructions[i], pop.Operands);
                                
                            }

                            i++;
                        }
                    }
                }
            }

            //If we have blocked cleanups, it is now safe to flush them
            if (VEM.PreventCleanup)
                VEM.PreventCleanup = false;
        }

        /// <summary>
        /// Releases all held resources
        /// </summary>
        /// <param name="disposing">True if called from the Dispose method, false if invoked from the finalizer</param>
        protected virtual void Dispose(bool disposing)
        {
            if (m_size > 0)
            {
                if (m_handle.IsAllocated)
                {
                    m_handle.Free();

                    if (m_externalData != PInvoke.cphvb_array_ptr.Null)
                        PInvoke.cphvb_data_set(m_externalData, IntPtr.Zero);
                }

                if (m_externalData != PInvoke.cphvb_array_ptr.Null)
                {
                    VEM.ExecuteRelease(new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_DESTROY, m_externalData));
                    m_externalData = PInvoke.cphvb_array_ptr.Null;
                }

                m_data = null;
                m_size = -1;

                if (disposing)
                    GC.SuppressFinalize(this);
            }

        }

        /// <summary>
        /// Releases all held resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        /// Destructor for non-disposed elements
        /// </summary>
        ~cphVBAccessor()
        {
            Dispose(false);
        }
    }
}
