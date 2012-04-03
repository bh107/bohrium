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

    public class ViewPtrKeeper : IDisposable
    {
        /// <summary>
        /// Instance of the VEM that is used
        /// </summary>
        protected static VEM VEM = NumCIL.cphVB.VEM.Instance;

        private PInvoke.cphvb_array_ptr m_ptr;
        public PInvoke.cphvb_array_ptr Pointer { get { return m_ptr; } }
        public ViewPtrKeeper(PInvoke.cphvb_array_ptr p)
        {
            m_ptr = p;
        }

        public void Dispose(bool disposing)
        {
            if (m_ptr != PInvoke.cphvb_array_ptr.Null)
            {
                VEM.ExecuteRelease(new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_DESTROY, m_ptr));
                m_ptr = PInvoke.cphvb_array_ptr.Null;
            }

            if (disposing)
                GC.SuppressFinalize(this);
        }

        public void Dispose() { Dispose(true); }
        ~ViewPtrKeeper() { Dispose(false); }
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
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_ABSOLUTE, "Abs");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_LOG, "Log");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_NEGATIVE, "Negate");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_EXP, "Exp");
            _opcode_func_name.Add(PInvoke.cphvb_opcode.CPHVB_POWER, "Pow");
        }

        /// <summary>
        /// Helper function to get the opcode mapping table for the current type
        /// </summary>
        /// <returns>A mapping between the type used for this executor and the cphVB opcodes</returns>
        public static Dictionary<Type, PInvoke.cphvb_opcode> CreateOpCodeMap<T>()
        {
            Dictionary<Type, PInvoke.cphvb_opcode> res = new Dictionary<Type, PInvoke.cphvb_opcode>();

            Type basic = typeof(NumCIL.Generic.NdArray<>);

            if (typeof(T) == typeof(sbyte))
                basic = typeof(NumCIL.Int8.NdArray);
            else if (typeof(T) == typeof(short))
                basic = typeof(NumCIL.Int16.NdArray);
            else if (typeof(T) == typeof(int))
                basic = typeof(NumCIL.Int32.NdArray);
            else if (typeof(T) == typeof(long))
                basic = typeof(NumCIL.Int64.NdArray);
            else if (typeof(T) == typeof(byte))
                basic = typeof(NumCIL.UInt8.NdArray);
            else if (typeof(T) == typeof(ushort))
                basic = typeof(NumCIL.UInt16.NdArray);
            else if (typeof(T) == typeof(uint))
                basic = typeof(NumCIL.UInt32.NdArray);
            else if (typeof(T) == typeof(ulong))
                basic = typeof(NumCIL.UInt64.NdArray);
            else if (typeof(T) == typeof(float))
                basic = typeof(NumCIL.Float.NdArray);
            else if (typeof(T) == typeof(double))
                basic = typeof(NumCIL.Double.NdArray);
            else
                throw new Exception("Unexpected type: " + (typeof(T)).FullName);

            foreach (var e in _opcode_func_name)
            {
                try { res[basic.Assembly.GetType(basic.Namespace + "." + e.Value)] = e.Key; }
                catch { }
            }

            res[typeof(NumCIL.CopyOp<T>)] = PInvoke.cphvb_opcode.CPHVB_IDENTITY;

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
    public class cphVBAccessor<T> : NumCIL.Generic.LazyAccessor<T>, IDisposable
    {
        /// <summary>
        /// Instance of the VEM that is used
        /// </summary>
        protected static VEM VEM = NumCIL.cphVB.VEM.Instance;

        /// <summary>
        /// The maximum number of instructions to queue
        /// </summary>
        protected static readonly long HIGH_WATER_MARK = 2000;

        /// <summary>
        /// A lookup table that maps NumCIL operation types to cphVB opcodes
        /// </summary>
        protected static Dictionary<Type, PInvoke.cphvb_opcode> OpcodeMap = OpCodeMapper.CreateOpCodeMap<T>();

        /// <summary>
        /// The view cache used to prevent repeated creation of views
        /// </summary>
        //protected static ViewCache<T> ViewCache = new ViewCache<T>(VEM);

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
        /// A pointer to internally allocated data which is pinned
        /// </summary>
        protected GCHandle m_handle;

        /// <summary>
        /// The default value to fill the array with
        /// </summary>
        protected T m_defaultValue;

        /// <summary>
        /// Returns the data block, flushed and updated
        /// </summary>
        public override T[] Data
        {
            get
            {
                MakeDataManaged();
                return base.Data;
            }
        }

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
        /// Creates a base array (a view of an array)
        /// </summary>
        /// <param name="size">The number of elements in the array</param>
        /// <param name="adr">An optional pointer to a pinned memory region</param>
        /// <returns>A new base array</returns>
        protected PInvoke.cphvb_array_ptr CreateBaseView(long size, IntPtr adr)
        {
            var v = VEM.CreateArray<T>(size);
            if (adr != IntPtr.Zero)
                v.Data = adr;
            return v;
        }

        /// <summary>
        /// Creates a new view of data
        /// </summary>
        /// <param name="shape">The shape to create the view for</param>
        /// <param name="baseArray">The array to set as base array</param>
        /// <returns>A new view</returns>
        protected PInvoke.cphvb_array_ptr CreateView(Shape shape, PInvoke.cphvb_array_ptr baseArray)
        {
            //Unroll, to avoid creating a Linq query for basic 3d shapes
            if (shape.Dimensions.Length == 1)
            {
                return VEM.CreateArray(
                    baseArray,
                    VEM.MapType(typeof(T)),
                    shape.Dimensions.Length,
                    (int)shape.Offset,
                    new long[] { shape.Dimensions[0].Length },
                    new long[] { shape.Dimensions[0].Stride },
                    false,
                    new PInvoke.cphvb_constant()
                );
            }
            else if (shape.Dimensions.Length == 2)
            {
                return VEM.CreateArray(
                    baseArray,
                    VEM.MapType(typeof(T)),
                    shape.Dimensions.Length,
                    (int)shape.Offset,
                    new long[] { shape.Dimensions[0].Length, shape.Dimensions[1].Length },
                    new long[] { shape.Dimensions[0].Stride, shape.Dimensions[1].Stride },
                    false,
                    new PInvoke.cphvb_constant()
                );
            }
            else if (shape.Dimensions.Length == 3)
            {
                return VEM.CreateArray(
                    baseArray,
                    VEM.MapType(typeof(T)),
                    shape.Dimensions.Length,
                    (int)shape.Offset,
                    new long[] { shape.Dimensions[0].Length, shape.Dimensions[1].Length, shape.Dimensions[2].Length },
                    new long[] { shape.Dimensions[0].Stride, shape.Dimensions[1].Stride, shape.Dimensions[2].Stride },
                    false,
                    new PInvoke.cphvb_constant()
                );
            }
            else
            {
                long[] lengths = new long[shape.Dimensions.LongLength];
                long[] strides = new long[shape.Dimensions.LongLength];
                for (int i = 0; i < lengths.LongLength; i++)
                {
                    var d = shape.Dimensions[i];
                    lengths[i] = d.Length;
                    strides[i] = d.Stride;
                }

                return VEM.CreateArray(
                    baseArray,
                    VEM.MapType(typeof(T)),
                    shape.Dimensions.Length,
                    (int)shape.Offset,
                    lengths,
                    strides,
                    false,
                    new PInvoke.cphvb_constant()
                );
            }
        }

        /// <summary>
        /// Pins the allocated data and returns the pinned pointer
        /// </summary>
        /// <returns>A pinned pointer</returns>
        protected PInvoke.cphvb_array_ptr Pin()
        {
            if (m_data == null && m_externalData == PInvoke.cphvb_array_ptr.Null)
            {
                //Data is not yet allocated, convert to external storage
                m_externalData = CreateBaseView(m_size, IntPtr.Zero);
                m_ownsData = false;
            }
            else if (m_externalData == PInvoke.cphvb_array_ptr.Null)
            {
                //Internally allocated data, we need to pin it
                if (!m_handle.IsAllocated)
                    m_handle = GCHandle.Alloc(m_data, GCHandleType.Pinned);

                m_externalData = CreateBaseView(m_size, m_handle.AddrOfPinnedObject());
                m_ownsData = true;
            }
            else if (m_ownsData && m_externalData.Data == IntPtr.Zero)
            {
                //Internally allocated data, we need to pin it
                if (!m_handle.IsAllocated)
                    m_handle = GCHandle.Alloc(m_data, GCHandleType.Pinned);
                m_externalData.Data = m_handle.AddrOfPinnedObject();
            }

            return m_externalData;
        }

        /// <summary>
        /// Unpins allocated data
        /// </summary>
        protected void Unpin()
        {
            if (m_externalData != PInvoke.cphvb_array_ptr.Null)
                VEM.Execute(new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_SYNC, m_externalData));

            if (m_handle.IsAllocated)
            {
                m_handle.Free();

                m_externalData.Data = IntPtr.Zero;
            }
        }


        /// <summary>
        /// Generates an unmanaged view pointer for the NdArray
        /// </summary>
        /// <param name="view">The NdArray to create the pointer for</param>
        /// <param name="createdViews">A list of already created views</param>
        /// <param name="createdBaseViews">A list of already created base views</param>
        /// <returns>An unmanaged view pointer</returns>
        protected PInvoke.cphvb_array_ptr CreateViewPtr(NdArray<T> view, Dictionary<IDataAccessor<T>, Tuple<PInvoke.cphvb_array_ptr, GCHandle>> createdBaseViews)
        {
            PInvoke.cphvb_array_ptr basep;

            if (view.m_data is cphVBAccessor<T>)
            {
                basep = ((cphVBAccessor<T>)view.m_data).Pin();
            }
            else
            {
                Tuple<PInvoke.cphvb_array_ptr, GCHandle> t;
                if (!createdBaseViews.TryGetValue(view.m_data, out t))
                {
                    GCHandle h = GCHandle.Alloc(view.m_data.Data, GCHandleType.Pinned);
                    t = new Tuple<PInvoke.cphvb_array_ptr,GCHandle>(CreateBaseView(view.m_data.Data.Length, h.AddrOfPinnedObject()), h);

                    createdBaseViews.Add(view.m_data, t);
                }

                basep = t.Item1;
            }

            if (view.Tag == null || ((ViewPtrKeeper)view.Tag).Pointer != basep)
                view.Tag = new ViewPtrKeeper(CreateView(view.Shape, basep));

            return ((ViewPtrKeeper)view.Tag).Pointer;
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
            T[] data = base.Data;

            //If data is allocated in cphVB, we need to flush it and de-allocate it
            if (m_externalData != PInvoke.cphvb_array_ptr.Null && !m_ownsData)
            {
                this.Unpin();

                //TODO: Figure out if we can avoid the copy by having a pinvoke method that returns a float[]

                IntPtr actualData = m_externalData.Data;
                if (actualData == IntPtr.Zero)
                {
                    if (!object.Equals(m_defaultValue, default(T)))
                    {
                        for (long i = 0; i < data.LongLength; i++)
                            data[i] = m_defaultValue;
                    }

                    //Otherwise the array has "empty" which will be zeroes in NumCIL
                }
                else
                {
                    if (m_size > int.MaxValue)
                        throw new OverflowException();

                    //Then copy the data into the local buffer
                    if (typeof(T) == typeof(float))
                        Marshal.Copy(actualData, (float[])(object)data, 0, (int)m_size);
                    else if (typeof(T) == typeof(double))
                        Marshal.Copy(actualData, (double[])(object)data, 0, (int)m_size);
                    else if (typeof(T) == typeof(sbyte))
                    {
                        //TODO: Probably faster to just call memcpy in native code
                        sbyte[] xref = (sbyte[])(object)data;
                        int sbytesize = Marshal.SizeOf(typeof(sbyte));

                        if (m_size > int.MaxValue)
                        {
                            IntPtr xptr = actualData;
                            for (long i = 0; i < m_size; i++)
                            {
                                xref[i] = (sbyte)Marshal.ReadByte(xptr);
                                xptr = IntPtr.Add(xptr, sbytesize);
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
                        //TODO: Probably faster to just call memcpy in native code
                        ushort[] xref = (ushort[])(object)data;
                        int ushortsize = Marshal.SizeOf(typeof(ushort));

                        if (m_size > int.MaxValue)
                        {
                            IntPtr xptr = actualData;
                            for (long i = 0; i < m_size; i++)
                            {
                                xref[i] = (ushort)Marshal.ReadInt16(xptr);
                                xptr = IntPtr.Add(xptr, ushortsize);
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
                        //TODO: Probably faster to just call memcpy in native code
                        uint[] xref = (uint[])(object)data;
                        int uintsize = Marshal.SizeOf(typeof(uint));

                        if (m_size > int.MaxValue)
                        {
                            IntPtr xptr = actualData;
                            for (long i = 0; i < m_size; i++)
                            {
                                xref[i] = (uint)Marshal.ReadInt32(xptr);
                                xptr = IntPtr.Add(xptr, uintsize);
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
                        //TODO: Probably faster to just call memcpy in native code
                        ulong[] xref = (ulong[])(object)data;
                        int ulongsize = Marshal.SizeOf(typeof(ulong));

                        if (m_size > int.MaxValue)
                        {
                            IntPtr xptr = actualData;
                            for (long i = 0; i < m_size; i++)
                            {
                                xref[i] = (ulong)Marshal.ReadInt64(xptr);
                                xptr = IntPtr.Add(xptr, ulongsize);
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


                //Release the unmanaged copy
                VEM.Execute(new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_DESTROY, m_externalData));
                m_externalData = PInvoke.cphvb_array_ptr.Null;
                m_ownsData = true;
            }
        }


        /// <summary>
        /// Executes all pending operations in the list
        /// </summary>
        /// <param name="work">The list of operations to execute</param>
        public override void ExecuteOperations(IEnumerable<PendingOperation<T>> work)
        {
            Dictionary<IDataAccessor<T>, Tuple<PInvoke.cphvb_array_ptr, GCHandle>> createdBaseViews = new Dictionary<IDataAccessor<T>, Tuple<PInvoke.cphvb_array_ptr, GCHandle>>();

            try
            {
                List<PendingOperation<T>> unsupported = new List<PendingOperation<T>>();
                List<PInvoke.cphvb_instruction> supported = new List<PInvoke.cphvb_instruction>();

                foreach (var op in work)
                {
                    Type t;
                    bool isScalar;
                    if (op.Operation is ScalarAccess<T>)
                    {
                        t = ((ScalarAccess<T>)op.Operation).Operation.GetType();
                        isScalar = true;
                    }
                    else if (op.Operation is GenerateOp<T>)
                    {
                        if (((cphVB.cphVBAccessor<T>)op.Operands[0].m_data).m_externalData != PInvoke.cphvb_array_ptr.Null || ((cphVB.cphVBAccessor<T>)op.Operands[0].m_data).m_ownsData)
                            throw new InvalidOperationException("Unexpected generate operation on already allocated array?");

                        T value = ((GenerateOp<T>)op.Operation).Value;
                        ((cphVB.cphVBAccessor<T>)op.Operands[0].m_data).m_defaultValue = value;
                        ((cphVB.cphVBAccessor<T>)op.Operands[0].m_data).m_externalData = VEM.CreateArray<T>(value, (int)op.Operands[0].m_data.Length);
                        continue;
                    }
                    else
                    {
                        t = op.Operation.GetType();
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
                            var scalarAcc = new cphVBAccessor<T>(1);
                            scalarAcc.m_defaultValue = ((ScalarAccess<T>)op.Operation).Value;
                            scalarAcc.m_externalData = VEM.CreateArray<T>(scalarAcc.m_defaultValue, 1);
                            
                            Shape bShape = Shape.ToBroadcastShapes(op.Operands[1].Shape, new Shape(1)).Item2;
                            var scalarOp = new NdArray<T>(scalarAcc, bShape);
                            
                            supported.Add(new PInvoke.cphvb_instruction(
                                opcode,
                                CreateViewPtr(op.Operands[0], createdBaseViews),
                                CreateViewPtr(scalarOp, createdBaseViews),
                                CreateViewPtr(op.Operands[1], createdBaseViews)
                            ));

                        }
                        else
                        {
                            if (op.Operands.Length == 1)
                                supported.Add(new PInvoke.cphvb_instruction(
                                    opcode,
                                    CreateViewPtr(op.Operands[0], createdBaseViews)
                                ));
                            else if (op.Operands.Length == 2)
                                supported.Add(new PInvoke.cphvb_instruction(
                                    opcode,
                                    CreateViewPtr(op.Operands[0], createdBaseViews),
                                    CreateViewPtr(op.Operands[1], createdBaseViews)
                                ));
                            else if (op.Operands.Length == 3)
                                supported.Add(new PInvoke.cphvb_instruction(
                                    opcode,
                                    CreateViewPtr(op.Operands[0], createdBaseViews),
                                    CreateViewPtr(op.Operands[1], createdBaseViews),
                                    CreateViewPtr(op.Operands[2], createdBaseViews)
                                ));
                            else
                                supported.Add(new PInvoke.cphvb_instruction(
                                    opcode,
                                    op.Operands.Select(x => CreateViewPtr(x, createdBaseViews))
                                ));
                        }
                    }
                    else
                    {
                        if (supported.Count > 0)
                        {
                            VEM.Execute(supported);
                            supported.Clear();
                        }

                        unsupported.Add(op);
                    }

                    if (op is IDisposable)
                        ((IDisposable)op).Dispose();
                }

                if (supported.Count > 0 && unsupported.Count > 0)
                    throw new InvalidOperationException("Unexpected result, both supported and non-supported operations");

                if (unsupported.Count > 0)
                    base.ExecuteOperations(unsupported);

                if (supported.Count > 0)
                    VEM.Execute(supported);
            }
            finally
            {
                foreach (var kp in createdBaseViews)
                {
                    kp.Value.Item1.Data = IntPtr.Zero;
                    kp.Value.Item2.Free();
                    VEM.ExecuteRelease(new PInvoke.cphvb_instruction(PInvoke.cphvb_opcode.CPHVB_DESTROY, kp.Value.Item1));
                }
            }
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
