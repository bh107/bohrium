using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;
using System.Runtime.InteropServices;

namespace NumCIL.cphVB
{
    /// <summary>
    /// Basic wrapper implementation of a cphvb VEM
    /// </summary>
    public class VEM : IDisposable
    {
        /// <summary>
        /// Singleton VEM instance
        /// </summary>
        private static VEM _instance = null;

        /// <summary>
        /// Accessor for singleton VEM instance
        /// </summary>
        public static VEM Instance
        {
            get
            {
                if (_instance == null)
                    _instance = new VEM();

                return _instance;
            }
        }

        /// <summary>
        /// Lock object for ensuring single threaded access to the VEM
        /// </summary>
        private object m_executelock = new object();

        /// <summary>
        /// Lock object for protecting the release queue
        /// </summary>
        private object m_releaselock = new object();

        /// <summary>
        /// ID for the user-defined reduce operation
        /// </summary>
        private readonly long m_reduceFunctionId;

        /// <summary>
        /// ID for the user-defined random operation
        /// </summary>
        private readonly long m_randomFunctionId;

        /// <summary>
        /// ID for the user-defined maxtrix multiplication operation
        /// </summary>
        private readonly long m_matmulFunctionId;

        /// <summary>
        /// A reference to the cphVB component for "self" aka the bridge
        /// </summary>
        private PInvoke.cphvb_component m_component;
        /// <summary>
        /// The unmanaged copy of the component
        /// </summary>
        private IntPtr m_componentPtr;
        /// <summary>
        /// A reference to the chpVB VEM
        /// </summary>
        private PInvoke.cphvb_component[] m_childs;
        /// <summary>
        /// The unmanaged copy of the childs array
        /// </summary>
        private IntPtr m_childsPtr;

        /// <summary>
        /// A list of cleanups not yet performed
        /// </summary>
        private List<IInstruction> m_cleanups = new List<IInstruction>();
        /// <summary>
        /// Flag that guards cleanup execution
        /// </summary>
        private bool m_preventCleanup = false;
        /// <summary>
        /// A ref-counter for base arrays
        /// </summary>
        private Dictionary<PInvoke.cphvb_array_ptr, List<ViewPtrKeeper>> m_baseArrayRefs = new Dictionary<PInvoke.cphvb_array_ptr, List<ViewPtrKeeper>>();
        /// <summary>
        /// Lookup table for all created userfunc structures
        /// </summary>
        private Dictionary<IntPtr, GCHandle> m_allocatedUserfuncs = new Dictionary<IntPtr, GCHandle>();
        /// <summary>
        /// GC Handles for managed data
        /// </summary>
        private Dictionary<PInvoke.cphvb_array_ptr, GCHandle> m_managedHandles = new Dictionary<PInvoke.cphvb_array_ptr, GCHandle>();

        /// <summary>
        /// Constructs a new VEM
        /// </summary>
        public VEM()
        {
//Disable "Unreachable code" warning
#pragma warning disable 0162
            if (cphvb_opcode.CPHVB_ADD == cphvb_opcode.CPHVB_SUBTRACT)
                throw new Exception("This version of NumCIL.cphVB contains invalid opcodes!");
#pragma warning restore

            m_component = PInvoke.cphvb_component_setup(out m_componentPtr);
            PInvoke.cphvb_error e = PInvoke.cphvb_component_children(m_component, out m_childs, out m_childsPtr);
            if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                throw new cphVBException(e);

            if (m_childs.Length > 1)
                throw new cphVBException(string.Format("Unexpected number of child nodes: {0}", m_childs.Length));

            e = m_childs[0].init(ref m_childs[0]);
            if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                throw new cphVBException(e);

            //The exception only happens with the debugger attached
            long id = 0;
            try
            {
                //Since the current implementation of reduce in cphvb is slow,
                // it can be disabled by an environment variable
                if (Environment.GetEnvironmentVariable("CPHVB_MANAGED_REDUCE") == null)
                {
                    e = m_childs[0].reg_func("cphvb_reduce", ref id);
                    if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                        id = 0;
                }
            }
            catch { id = 0; }
            m_reduceFunctionId = id;

            id = 0;
            try
            {
                e = m_childs[0].reg_func("cphvb_random", ref id);
                if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                    id = 0;
            }
            catch { id = 0; }
            m_randomFunctionId = id;

            id = 0;
            try
            {
                e = m_childs[0].reg_func("cphvb_matmul", ref id);
                if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                    id = 0;
            }
            catch { id = 0; }
            m_matmulFunctionId = id;
        }

        /// <summary>
        /// Gets or sets a value that determines if cleanup execution is currently disabled
        /// </summary>
        public bool PreventCleanup
        {
            get { return m_preventCleanup; }
            set
            {
                m_preventCleanup = value;
                if (!m_preventCleanup)
                    ExecuteCleanups();
            }
        }

        /// <summary>
        /// Invokes garbage collection and flushes all pending cleanup messages
        /// </summary>
        public void Flush()
        {
            GC.Collect();
            ExecuteCleanups();
        }

        /// <summary>
        /// Executes a list of instructions
        /// </summary>
        /// <param name="insts"></param>
        public void Execute(params IInstruction[] insts)
        {
            Execute((IEnumerable<IInstruction>)insts);
        }

        /// <summary>
        /// Registers instructions for later execution, usually destroy calls
        /// </summary>
        /// <param name="insts">The instructions to queue</param>
        public void ExecuteRelease(params IInstruction[] insts)
        {
            //Locks are re-entrant, so we lock here to enforce order
            lock(m_releaselock)
                foreach (var i in insts)
                    ExecuteRelease((PInvoke.cphvb_instruction)i);
        }

        public void ExecuteRelease(PInvoke.cphvb_array_ptr array, GCHandle handle)
        {
            lock (m_releaselock)
            {
                System.Diagnostics.Debug.Assert(array.BaseArray == PInvoke.cphvb_array_ptr.Null);
                if (handle.IsAllocated)
                    m_managedHandles.Add(array, handle);
                ExecuteRelease(new PInvoke.cphvb_instruction(cphvb_opcode.CPHVB_DISCARD, array));
            }
        }




        /// <summary>
        /// Registers an instruction for later execution, usually destroy calls
        /// </summary>
        /// <param name="inst">The instruction to queue</param>
        public void ExecuteRelease(PInvoke.cphvb_instruction inst)
        {
            lock (m_releaselock)
            {
                if (inst.opcode == cphvb_opcode.CPHVB_DISCARD)
                {
                    var ar = inst.operand0;
                    if (ar.BaseArray != PInvoke.cphvb_array_ptr.Null)
                    {
                        var lst = m_baseArrayRefs[ar.BaseArray];
                        for (int i = lst.Count - 1; i >= 0; i--)
                            if (lst[i].Pointer == ar)
                                lst.RemoveAt(i);

                    }
                    else
                    {
                        var lst = m_baseArrayRefs[ar];
                        while(lst.Count > 0)
                            lst[0].Dispose();
                        
                        m_baseArrayRefs.Remove(ar);
                    }

                    //Discard the view
                    m_cleanups.Add(inst);
                }
                else
                {
                    m_cleanups.Add(inst);
                }
            }
        }

        /// <summary>
        /// Executes a list of instructions
        /// </summary>
        /// <param name="inst_list">The list of instructions to execute</param>
        public void Execute(IEnumerable<IInstruction> inst_list)
        {
            var lst = inst_list;
            List<IInstruction> cleanup_lst = null;
            List<Tuple<long, PInvoke.cphvb_instruction, GCHandle>> handles = null;

            if (!m_preventCleanup && m_cleanups.Count > 0)
            {
                lock (m_releaselock)
                {
                    cleanup_lst = System.Threading.Interlocked.Exchange(ref m_cleanups, new List<IInstruction>());

                    GCHandle tmp;
                    long ix = inst_list.LongCount();
                    foreach (PInvoke.cphvb_instruction inst in cleanup_lst)
                    {
                        if (inst.opcode == cphvb_opcode.CPHVB_DISCARD && m_managedHandles.TryGetValue(inst.operand0, out tmp))
                        {
                            if (handles == null)
                                handles = new List<Tuple<long, PInvoke.cphvb_instruction, GCHandle>>();
                            handles.Add(new Tuple<long, PInvoke.cphvb_instruction, GCHandle>(ix, inst, tmp));
                        }
                        ix++;
                    }

                    lst = lst.Concat(cleanup_lst);
                }
            }

            long errorIndex = -1;

            try
            {
                lock (m_executelock)
                    ExecuteWithoutLocks(lst, out errorIndex);
            }
            catch
            {
                //This catch handler protects against leaks that happen during execution
                if (cleanup_lst != null)
                    lock (m_releaselock)
                    {
                        errorIndex -= inst_list.LongCount();
                        if (errorIndex > 0)
                        {
                            cleanup_lst.RemoveRange(0, (int)errorIndex);
                            cleanup_lst.AddRange(m_cleanups);
                            System.Threading.Interlocked.Exchange(ref m_cleanups, cleanup_lst);
                        }
                    }

                throw;
            }
            finally
            {
                if (handles != null)
                    lock (m_releaselock)
                    {
                        foreach (var kp in handles)
                        {
                            if (errorIndex == -1 || kp.Item1 < errorIndex)
                            {
                                m_managedHandles.Remove(kp.Item2.operand0);
                                kp.Item3.Free();
                            }
                        }
                    }
            }
        }

        /// <summary>
        /// Executes all pending cleanup instructions
        /// </summary>
        public void ExecuteCleanups()
        {
            if (!m_preventCleanup && m_cleanups.Count > 0)
            {
				//Atomically reset instruction list and get copy
                List<IInstruction> lst;
                lock (m_releaselock)
				    lst = System.Threading.Interlocked.Exchange (ref m_cleanups, new List<IInstruction>());

                long errorIndex = -1;
                try
                {
                    lock (m_executelock)
                        ExecuteWithoutLocks(lst, out errorIndex);
                }
                catch
                {
                    lock (m_releaselock)
                    {
                        lst.RemoveRange(0, (int)errorIndex);
                        lst.AddRange(m_cleanups);
                        System.Threading.Interlocked.Exchange(ref m_cleanups, lst);
                    }
                }
            }
        }

        /// <summary>
        /// Reshuffles instructions to honor cphVB rules
        /// </summary>
        /// <param name="list">The list of instructions to reshuffle</param>
        private void ReshuffleInstructions(PInvoke.cphvb_instruction[] list)
        {
            if (list.LongLength <= 1)
                return;

            long lastIx = list.LongLength;
            for(long i = 0; i < lastIx; i++)
            {
                var inst = list[i];
                if (inst.opcode == cphvb_opcode.CPHVB_DISCARD && inst.operand0.BaseArray == PInvoke.cphvb_array_ptr.Null)
                {
                    Console.WriteLine("Shuffling list, i: {0}, inst: {1}, lastIx: {2}", i, inst, lastIx);
                    lastIx--;
                    var tmp = list[lastIx];
                    list[lastIx] = inst;
                    list[i] = tmp;
                }
            }
        }

        /// <summary>
        /// Internal execution handler, runs without locking of any kind
        /// </summary>
        /// <param name="inst_list">The list of instructions to execute</param>
        private void ExecuteWithoutLocks(IEnumerable<IInstruction> inst_list, out long errorIndex)
        {
            List<GCHandle> cleanups = new List<GCHandle>();
            long destroys = 0;
            errorIndex = -1;

            try
            {
                PInvoke.cphvb_instruction[] instrBuffer = inst_list.Select(x => (PInvoke.cphvb_instruction)x).ToArray();
                //ReshuffleInstructions(instrBuffer);

                foreach (var inst in instrBuffer)
                {
                    if (inst.opcode == cphvb_opcode.CPHVB_DISCARD)
                        destroys++;
                    if (inst.userfunc != IntPtr.Zero)
                    {
                        cleanups.Add(m_allocatedUserfuncs[inst.userfunc]);
                        m_allocatedUserfuncs.Remove(inst.userfunc);
                    }
                }

                PInvoke.cphvb_error e = m_childs[0].execute(instrBuffer.LongLength, instrBuffer);

                if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                {
                    if (e == PInvoke.cphvb_error.CPHVB_PARTIAL_SUCCESS)
                    {
                        for (long i = 0; i < instrBuffer.LongLength; i++)
                        {
                            if (instrBuffer[i].status == PInvoke.cphvb_error.CPHVB_INST_NOT_SUPPORTED)
                            {
                                errorIndex = i;
                                throw new cphVBNotSupportedInstruction(instrBuffer[i].opcode, i);
                            }

                            if (instrBuffer[i].status != PInvoke.cphvb_error.CPHVB_SUCCESS)
                            {
                                errorIndex = i;
                                break;
                            }
                        }
                    }

                    throw new cphVBException(e);
                }

            }
            finally
            {
                foreach (var h in cleanups)
                    h.Free();
            }
        }

        /// <summary>
        /// Creates a cphvb descriptor for a base array, without assigning the actual data
        /// </summary>
        /// <param name="d">The array to map</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.cphvb_array_ptr CreateBaseArray(Array d)
        {
            return CreateBaseArray(MapType(d.GetType().GetElementType()), d.LongLength);
        }

        /// <summary>
        /// Creates a base array with uninitialized memory
        /// </summary>
        /// <typeparam name="T">The data type for the array</typeparam>
        /// <param name="size">The size of the generated base array</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.cphvb_array_ptr CreateBaseArray<T>(long size)
        {
            return CreateBaseArray(MapType(typeof(T)), size);
        }

        public static PInvoke.cphvb_type MapType(Type t)
        {
            if (t == typeof(bool))
                return PInvoke.cphvb_type.CPHVB_BOOL;
            else if (t == typeof(sbyte))
                return PInvoke.cphvb_type.CPHVB_INT8;
            else if (t == typeof(short))
                return PInvoke.cphvb_type.CPHVB_INT16;
            else if (t == typeof(int))
                return PInvoke.cphvb_type.CPHVB_INT32;
            else if (t == typeof(long))
                return PInvoke.cphvb_type.CPHVB_INT64;
            else if (t == typeof(byte))
                return PInvoke.cphvb_type.CPHVB_UINT8;
            else if (t == typeof(ushort))
                return PInvoke.cphvb_type.CPHVB_UINT16;
            else if (t == typeof(uint))
                return PInvoke.cphvb_type.CPHVB_UINT32;
            else if (t == typeof(ulong))
                return PInvoke.cphvb_type.CPHVB_UINT64;
            else if (t == typeof(float))
                return PInvoke.cphvb_type.CPHVB_FLOAT32;
            else if (t == typeof(double))
                return PInvoke.cphvb_type.CPHVB_FLOAT64;
            else
                throw new cphVBException(string.Format("Unsupported data type: " + t.FullName));
        }

        /// <summary>
        /// Creates a cphvb base array or view descriptor
        /// </summary>
        /// <param name="type">The cphvb type of data in the array</param>
        /// <param name="size">The size of the base array</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.cphvb_array_ptr CreateBaseArray(PInvoke.cphvb_type type, long size)
        {
            var ptr = CreateArray(
                PInvoke.cphvb_array_ptr.Null,
                type,
                1,
                0,
                new long[] { size },
                new long[] { 1 }
            );

            lock (m_releaselock)
                m_baseArrayRefs.Add(ptr, new List<ViewPtrKeeper>());

            return ptr;
        }

        /// <summary>
        /// Creates a cphvb view descriptor
        /// </summary>
        /// <param name="basearray">The base array pointer</param>
        /// <param name="type">The cphvb type of data in the array</param>
        /// <param name="ndim">Number of dimensions</param>
        /// <param name="start">The offset into the base array</param>
        /// <param name="shape">The shape values for each dimension</param>
        /// <param name="stride">The stride values for each dimension</param>
        /// <returns>The pointer to the array descriptor</returns>
        public ViewPtrKeeper CreateView(PInvoke.cphvb_array_ptr basearray, PInvoke.cphvb_type type, long ndim, long start, long[] shape, long[] stride)
        {
            if (basearray == PInvoke.cphvb_array_ptr.Null)
                throw new ArgumentException("Base array cannot be null for a view");
            var ptr = new ViewPtrKeeper(CreateArray(basearray, type, ndim, start, shape, stride));
            lock (m_releaselock)
                m_baseArrayRefs[basearray].Add(ptr);

            return ptr;
        }

        /// <summary>
        /// Creates a cphvb base array or view descriptor
        /// </summary>
        /// <param name="basearray">The base array pointer if creating a view or IntPtr.Zero if the view is a base array</param>
        /// <param name="type">The cphvb type of data in the array</param>
        /// <param name="ndim">Number of dimensions</param>
        /// <param name="start">The offset into the base array</param>
        /// <param name="shape">The shape values for each dimension</param>
        /// <param name="stride">The stride values for each dimension</param>
        /// <returns>The pointer to the array descriptor</returns>
        protected PInvoke.cphvb_array_ptr CreateArray(PInvoke.cphvb_array_ptr basearray, PInvoke.cphvb_type type, long ndim, long start, long[] shape, long[] stride)
        {
            PInvoke.cphvb_error e;
            PInvoke.cphvb_array_ptr res;
            lock (m_executelock)
            {
                e = m_childs[0].create_array(basearray, type, ndim, start, shape, stride, out res);
            }

            if (e == PInvoke.cphvb_error.CPHVB_OUT_OF_MEMORY)
            {
                //If we get this, it can be because some of the unmanaged views are still kept in memory
                Console.WriteLine("Ouch, forcing GC, allocated views: {0}", m_baseArrayRefs.Count + m_baseArrayRefs.Values.Select(x => x.Count).Sum());
                GC.Collect();
                ExecuteCleanups();

                lock (m_executelock)
                    e = m_childs[0].create_array(basearray, type, ndim, start, shape, stride, out res);
            }

            if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                throw new cphVBException(e);

            return res;
        }

        /// <summary>
        /// Releases all resources held
        /// </summary>
        public void Dispose()
        {
            //Ensure all views are collected
            GC.Collect();

            if (m_baseArrayRefs.Count > 0)
            {
                Console.WriteLine("WARNING: Found allocated views on shutdown");
                foreach (var k in m_baseArrayRefs.Values.ToArray())
                    foreach (var m in k.ToArray())
                        m.Dispose();

                GC.Collect();
            }

            if (m_baseArrayRefs.Count > 0)
                throw new Exception("Some base arrays were stil allocated during VEM shutdown");

            m_preventCleanup = false;
            ExecuteCleanups();

            lock (m_executelock)
            {
                if (m_childs != null)
                {
                    for (int i = 0; i < m_childs.Length; i++)
                        m_childs[i].shutdown();

                    if (m_childsPtr != IntPtr.Zero)
                    {
                        for (int i = 0; i < m_childs.Length; i++)
                        {
                            IntPtr cur = Marshal.ReadIntPtr(m_childsPtr, Marshal.SizeOf(typeof(IntPtr)) * i);
                            PInvoke.cphvb_component_free(cur);
                            cur += Marshal.SizeOf(typeof(IntPtr));
                        }

                        m_childsPtr = IntPtr.Zero;
                    }

                    m_childs = null;
                }

                if (m_componentPtr != IntPtr.Zero)
                {
                    PInvoke.cphvb_component_free(m_componentPtr);
                    m_component.config = IntPtr.Zero;
                }

                m_componentPtr = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Finalizes the VEM and shuts down the cphVB components
        /// </summary>
        ~VEM()
        {
            Dispose();
        }

        /// <summary>
        /// Generates an unmanaged view pointer for the NdArray
        /// </summary>
        /// <param name="view">The NdArray to create the pointer for</param>
        /// <returns>An unmanaged view pointer</returns>
        protected ViewPtrKeeper CreateViewPtr<T>(NdArray<T> view)
        {
            return CreateViewPtr<T>(MapType(typeof(T)), view);
        }

        /// <summary>
        /// Generates an unmanaged view pointer for the NdArray
        /// </summary>
        /// <param name="type">The type of data</param>
        /// <param name="view">The NdArray to create the pointer for</param>
        /// <typeparam name="T">The datatype in the view</typeparam>
        /// <returns>An unmanaged view pointer</returns>
        protected ViewPtrKeeper CreateViewPtr<T>(PInvoke.cphvb_type type, NdArray<T> view)
        {
            PInvoke.cphvb_array_ptr basep;

            if (view.DataAccessor is cphVBAccessor<T>)
            {
                basep = ((cphVBAccessor<T>)view.DataAccessor).BaseArrayPtr;
            }
            else
            {
                if (view.DataAccessor.Tag is ViewPtrKeeper)
                {
                    basep = ((ViewPtrKeeper)view.DataAccessor.Tag).Pointer;
                }
                else
                {
                    GCHandle h = GCHandle.Alloc(view.DataAccessor.AsArray(), GCHandleType.Pinned);
                    basep = CreateBaseArray<T>(view.DataAccessor.AsArray().Length);
                    basep.Data = h.AddrOfPinnedObject();
                    view.DataAccessor.Tag = new ViewPtrKeeper(basep, h);
                }
            }

            if (view.Tag as ViewPtrKeeper == null || ((ViewPtrKeeper)view.Tag).Pointer == PInvoke.cphvb_array_ptr.Null || ((ViewPtrKeeper)view.Tag).Pointer.BaseArray != basep)
                view.Tag = CreateView(type, view.Shape, basep);

            return (ViewPtrKeeper)view.Tag;
        }

        /// <summary>
        /// Creates a new view of data
        /// </summary>
        /// <param name="shape">The shape to create the view for</param>
        /// <param name="baseArray">The array to set as base array</param>
        /// <typeparam name="T">The type of data in the view</typeparam>
        /// <returns>A new view</returns>
        public ViewPtrKeeper CreateView<T>(Shape shape, PInvoke.cphvb_array_ptr baseArray)
        {
            return CreateView(MapType(typeof(T)), shape, baseArray);
        }

        /// <summary>
        /// Creates a new view of data
        /// </summary>
        /// <param name="CPHVB_TYPE">The data type of the view</param>
        /// <param name="shape">The shape to create the view for</param>
        /// <param name="baseArray">The array to set as base array</param>
        /// <returns>A new view</returns>
        public ViewPtrKeeper CreateView(PInvoke.cphvb_type CPHVB_TYPE, Shape shape, PInvoke.cphvb_array_ptr baseArray)
        {
            //Unroll, to avoid creating a Linq query for basic 3d shapes
            switch(shape.Dimensions.Length)
            {
                case 1:
                    return CreateView(
                        baseArray,
                        CPHVB_TYPE,
                        shape.Dimensions.Length,
                        (int)shape.Offset,
                        new long[] { shape.Dimensions[0].Length },
                        new long[] { shape.Dimensions[0].Stride }
                    );
                case 2:
                    return CreateView(
                            baseArray,
                            CPHVB_TYPE,
                            shape.Dimensions.Length,
                            (int)shape.Offset,
                            new long[] { shape.Dimensions[0].Length, shape.Dimensions[1].Length },
                            new long[] { shape.Dimensions[0].Stride, shape.Dimensions[1].Stride }
                        );
                case 3:
                    return CreateView(
                        baseArray,
                        CPHVB_TYPE,
                        shape.Dimensions.Length,
                        (int)shape.Offset,
                        new long[] { shape.Dimensions[0].Length, shape.Dimensions[1].Length, shape.Dimensions[2].Length },
                        new long[] { shape.Dimensions[0].Stride, shape.Dimensions[1].Stride, shape.Dimensions[2].Stride }
                    );
                default:
                    long[] lengths = new long[shape.Dimensions.LongLength];
                    long[] strides = new long[shape.Dimensions.LongLength];
                    for (int i = 0; i < lengths.LongLength; i++)
                    {
                        var d = shape.Dimensions[i];
                        lengths[i] = d.Length;
                        strides[i] = d.Stride;
                    }

                    return CreateView(
                        baseArray,
                        CPHVB_TYPE,
                        shape.Dimensions.Length,
                        (int)shape.Offset,
                        lengths,
                        strides
                    );
            }
        }

        public IInstruction CreateInstruction<T>(cphvb_opcode opcode, NdArray<T> operand, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
        {
            return CreateInstruction<T>(MapType(typeof(T)), opcode, operand);
        }
        public IInstruction CreateInstruction<T>(cphvb_opcode opcode, NdArray<T> op1, NdArray<T> op2, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
        {
            return CreateInstruction<T>(MapType(typeof(T)), opcode, op1, op2);
        }
        public IInstruction CreateInstruction<T>(cphvb_opcode opcode, NdArray<T> op1, NdArray<T> op2, NdArray<T> op3, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
        {
            return CreateInstruction<T>(MapType(typeof(T)), opcode, op1, op2, op3);
        }
        public IInstruction CreateInstruction<T>(cphvb_opcode opcode, IEnumerable<NdArray<T>> operands, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
        {
            return CreateInstruction<T>(MapType(typeof(T)), opcode, operands);
        }

        public IInstruction CreateInstruction<T>(PInvoke.cphvb_type type, cphvb_opcode opcode, NdArray<T> operand, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
        {
            return new PInvoke.cphvb_instruction(opcode, CreateViewPtr<T>(type, operand).Pointer, constant);
        }

        public IInstruction CreateInstruction<T>(PInvoke.cphvb_type type, cphvb_opcode opcode, NdArray<T> op1, NdArray<T> op2, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
        {
            return new PInvoke.cphvb_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, CreateViewPtr<T>(type, op2).Pointer, constant);
        }

        public IInstruction CreateInstruction<T>(PInvoke.cphvb_type type, cphvb_opcode opcode, NdArray<T> op1, PInvoke.cphvb_constant constant, NdArray<T> op2)
        {
            return new PInvoke.cphvb_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, constant, CreateViewPtr<T>(type, op2).Pointer);
        }

        public IInstruction CreateInstruction<T>(PInvoke.cphvb_type type, cphvb_opcode opcode, NdArray<T> op1, NdArray<T> op2, NdArray<T> op3, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
        {
            return new PInvoke.cphvb_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, CreateViewPtr<T>(type, op2).Pointer, CreateViewPtr<T>(type, op3).Pointer, constant);
        }

        public IInstruction CreateInstruction<T>(PInvoke.cphvb_type type, cphvb_opcode opcode, IEnumerable<NdArray<T>> operands, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
        {
            return new PInvoke.cphvb_instruction(opcode, operands.Select(x => CreateViewPtr<T>(type, x).Pointer), constant);
        }

		public IInstruction CreateConversionInstruction<Ta, Tb>(PInvoke.cphvb_type typea, NdArray<Ta> output, NdArray<Tb> input)
        {
            return new PInvoke.cphvb_instruction(NumCIL.cphVB.cphvb_opcode.CPHVB_IDENTITY, CreateViewPtr<Ta>(typea, output).Pointer, CreateViewPtr<Tb>(input).Pointer);
        }

        public IInstruction ReCreateInstruction<T>(PInvoke.cphvb_type type, IInstruction instruction, IEnumerable<NdArray<T>> operands)
        {
            if (instruction is PInvoke.cphvb_instruction)
                return new PInvoke.cphvb_instruction(instruction.OpCode, operands.Select(x => CreateViewPtr<T>(type, x).Pointer), ((PInvoke.cphvb_instruction)instruction).constant);
            else
                throw new Exception("Unknown instruction type");
        }

        public IInstruction CreateRandomInstruction<T>(PInvoke.cphvb_type type, NdArray<T> op1)
        {
            if (!SupportsRandom)
                throw new cphVBException("The VEM/VE setup does not support the random function");

            if (op1.Shape.Offset != 0 || !op1.Shape.IsPlain || op1.Shape.Elements != op1.DataAccessor.Length)
                throw new Exception("The shape of the element that is sent to the random implementation must be a non-shape plain array");

            GCHandle gh = GCHandle.Alloc(
                new PInvoke.cphvb_userfunc_random(
                    m_randomFunctionId,
                    CreateViewPtr<T>(type, op1).Pointer.BaseArray
                ), 
                GCHandleType.Pinned
            );

            IntPtr adr = gh.AddrOfPinnedObject();

            m_allocatedUserfuncs.Add(adr, gh);

            return new PInvoke.cphvb_instruction(
                cphvb_opcode.CPHVB_USERFUNC,
                adr                    
            );
        }

        public IInstruction CreateReduceInstruction<T>(PInvoke.cphvb_type type, cphvb_opcode opcode, long axis, NdArray<T> op1, NdArray<T>op2)
        {
            if (!SupportsReduce)
                throw new cphVBException("The VEM/VE setup does not support the reduce function");

            GCHandle gh = GCHandle.Alloc(
                new PInvoke.cphvb_userfunc_reduce(
                    m_reduceFunctionId,
                    opcode,
                    axis,
                    CreateViewPtr<T>(type, op1).Pointer,
                    CreateViewPtr<T>(type, op2).Pointer
                ), 
                GCHandleType.Pinned
            );

            IntPtr adr = gh.AddrOfPinnedObject();

            m_allocatedUserfuncs.Add(adr, gh);

            return new PInvoke.cphvb_instruction(
                cphvb_opcode.CPHVB_USERFUNC,
                adr
            );
        }

        public IInstruction CreateMatmulInstruction<T>(PInvoke.cphvb_type type, NdArray<T> op1, NdArray<T> op2, NdArray<T> op3)
        {
            if (!SupportsMatmul)
                throw new cphVBException("The VEM/VE setup does not support the matmul function");

            GCHandle gh = GCHandle.Alloc(
                new PInvoke.cphvb_userfunc_matmul(
                    m_matmulFunctionId,
                    CreateViewPtr<T>(type, op1).Pointer,
                    CreateViewPtr<T>(type, op2).Pointer,
                    CreateViewPtr<T>(type, op3).Pointer
                ),
                GCHandleType.Pinned
            );

            IntPtr adr = gh.AddrOfPinnedObject();

            m_allocatedUserfuncs.Add(adr, gh);

            return new PInvoke.cphvb_instruction(
                cphvb_opcode.CPHVB_USERFUNC,
                adr
            );
        }

        /// <summary>
        /// Gets a value indicating if the Reduce operation is supported
        /// </summary>
        public bool SupportsReduce { get { return m_reduceFunctionId > 0; } }
        //public bool SupportsReduce { get { return false; } }

        /// <summary>
        /// Gets a value indicating if the Random operation is supported
        /// </summary>
        public bool SupportsRandom { get { return m_randomFunctionId > 0; } }

        /// <summary>
        /// Gets a value indicating if the Matrix Multiplication operation is supported
        /// </summary>
        public bool SupportsMatmul { get { return m_matmulFunctionId > 0; } }
    }
}
