using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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
        /// Lock object for ensuring single threaded access to the VEM
        /// </summary>
        private object m_executelock = new object();

        /// <summary>
        /// Lock object for protecting the release queue
        /// </summary>
        private object m_releaselock = new object();

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
        /// A reference to the cphVB component for "self" aka the bridge
        /// </summary>
        private PInvoke.cphvb_com m_component;
        /// <summary>
        /// A reference to the chpVB VEM
        /// </summary>
        private PInvoke.cphvb_com[] m_childs;

        /// <summary>
        /// A list of cleanups not yet performed
        /// </summary>
        private List<PInvoke.cphvb_instruction> m_cleanups = new List<PInvoke.cphvb_instruction>();

        /// <summary>
        /// Constructs a new VEM
        /// </summary>
        public VEM()
        {
            m_component = PInvoke.cphvb_com_setup();
            PInvoke.cphvb_error e = PInvoke.cphvb_com_children(m_component, out m_childs);
            if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                throw new cphVBException(e);

            if (m_childs.Length > 1)
                throw new cphVBException(string.Format("Unexpected number of child nodes: {0}", m_childs.Length));

            e = m_childs[0].init(ref m_childs[0]);
            if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                throw new cphVBException(e);
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
        public void Execute(params PInvoke.cphvb_instruction[] insts)
        {
            Execute((IEnumerable<PInvoke.cphvb_instruction>)insts);
        }

        /// <summary>
        /// Registers instructions for later execution, usually destroy calls
        /// </summary>
        /// <param name="insts">The instructions to queue</param>
        public void ExecuteRelease(params PInvoke.cphvb_instruction[] insts)
        {
            //Lock is not really required as the GC is single threaded,
            // but user code could also call this
            lock(m_releaselock)
                m_cleanups.AddRange(insts);
        }

        /// <summary>
        /// Registers an instruction for later execution, usually destroy calls
        /// </summary>
        /// <param name="inst">The instruction to queue</param>
        public void ExecuteRelease(PInvoke.cphvb_instruction inst)
        {
            //Lock is not really required as the GC is single threaded,
            // but user code could also call this
            lock (m_releaselock)
                m_cleanups.Add(inst);
        }

        /// <summary>
        /// Executes a list of instructions
        /// </summary>
        /// <param name="inst_list">The list of instructions to execute</param>
        public void Execute(IEnumerable<PInvoke.cphvb_instruction> inst_list)
        {
            lock (m_executelock)
                UnprotectedExecute(inst_list);

            ExecuteCleanups();
        }

        /// <summary>
        /// Executes all pending cleanup instructions
        /// </summary>
        private void ExecuteCleanups()
        {
            if (m_cleanups.Count > 0)
            {
                //Lock free swapping, ensures that we never block the garbage collector
                List<PInvoke.cphvb_instruction> lst = m_cleanups;
                m_cleanups = new List<PInvoke.cphvb_instruction>();

                lock (m_executelock)
                    UnprotectedExecute(lst);
            }
        }

        /// <summary>
        /// Internal execution handler, runs without locking of any kind
        /// </summary>
        /// <param name="inst_list">The list of instructions to execute</param>
        private void UnprotectedExecute(IEnumerable<PInvoke.cphvb_instruction> inst_list)
        {
            //We need to execute multiple times if we have more than CPHVB_MAX_NO_INST instructions
            PInvoke.cphvb_instruction[] buf = new PInvoke.cphvb_instruction[PInvoke.CPHVB_MAX_NO_INST];

            int i = 0;
            foreach (PInvoke.cphvb_instruction inst in inst_list)
            {
                buf[i++] = inst;
                if (i >= buf.Length)
                {
                    PInvoke.cphvb_error e = m_childs[0].execute(buf.LongLength, buf);

                    if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                        throw new cphVBException(e);
                    i = 0;
                }
            }

            if (i != 0)
            {
                PInvoke.cphvb_error e = m_childs[0].execute(i, buf);

                if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                    throw new cphVBException(e);
            }

        }

        /// <summary>
        /// Creates a cphvb descriptor for a base array, without assigning the actual data
        /// </summary>
        /// <param name="d">The array to map</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.cphvb_array_ptr CreateArray(Array d)
        {
            return CreateArray(
                PInvoke.cphvb_array_ptr.Null,
                MapType(d.GetType().GetElementType()),
                1,
                0,
                new long[] { d.Length },
                new long[] { 1 },
                false,
                new PInvoke.cphvb_constant() { uint64 = 0 }
                );
        }

        /// <summary>
        /// Creates a base array from a scalar/initial value
        /// </summary>
        /// <typeparam name="T">The data type for the array</typeparam>
        /// <param name="data">The initial value for the base array</param>
        /// <param name="size">The size of the generated base array</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.cphvb_array_ptr CreateArray<T>(T data, int size)
        {
            return CreateArray(
                PInvoke.cphvb_array_ptr.Null,
                MapType(typeof(T)),
                1,
                0,
                new long[] { size },
                new long[] { 1 },
                true,
                new PInvoke.cphvb_constant().Set(data)
                );
        }

        /// <summary>
        /// Creates a base array with uninitialized memory
        /// </summary>
        /// <typeparam name="T">The data type for the array</typeparam>
        /// <param name="size">The size of the generated base array</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.cphvb_array_ptr CreateArray<T>(long size)
        {
            return CreateArray(
                PInvoke.cphvb_array_ptr.Null,
                MapType(typeof(T)),
                1,
                0,
                new long[] { size },
                new long[] { 1 },
                false,
                new PInvoke.cphvb_constant()
                );
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
        /// <param name="basearray">The base array pointer if creating a view or IntPtr.Zero if the view is a base array</param>
        /// <param name="type">The cphvb type of data in the array</param>
        /// <param name="ndim">Number of dimensions</param>
        /// <param name="start">The offset into the base array</param>
        /// <param name="shape">The shape values for each dimension</param>
        /// <param name="stride">The stride values for each dimension</param>
        /// <param name="has_init_value">A value indicating if the data has a initial value</param>
        /// <param name="init_value">The initial value if any</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.cphvb_array_ptr CreateArray(PInvoke.cphvb_array_ptr basearray, PInvoke.cphvb_type type, long ndim, long start, long[] shape, long[] stride, bool has_init_value, PInvoke.cphvb_constant init_value)
        {
            PInvoke.cphvb_error e;
            PInvoke.cphvb_array_ptr res;
            lock (m_executelock)
            {
                e = m_childs[0].create_array(basearray, type, ndim, start, shape, stride, has_init_value ? 1 : 0, init_value, out res);
            }

            if (e == PInvoke.cphvb_error.CPHVB_OUT_OF_MEMORY)
            {
                //If we get this, it can be because some of the unmanaged views are still kept in memory
                GC.Collect();
                ExecuteCleanups();

                lock (m_executelock)
                    e = m_childs[0].create_array(basearray, type, ndim, start, shape, stride, has_init_value ? 1 : 0, init_value, out res);
            }

            if (e != PInvoke.cphvb_error.CPHVB_SUCCESS)
                throw new cphVBException(e);

            System.Threading.Interlocked.Increment(ref m_allocatedviews);

            return res;
        }

        /// <summary>
        /// Releases all resources held
        /// </summary>
        public void Dispose()
        {
            ExecuteCleanups();

            //TODO: Probably not good because the call will "free" the component as well, and that is semi-managed
            lock (m_executelock)
            {
                if (m_childs != null)
                {
                    for (int i = 0; i < m_childs.Length; i++)
                        PInvoke.cphvb_com_free(ref m_childs[i]);

                    m_childs = null;
                }

                if (m_component.config != IntPtr.Zero)
                {
                    PInvoke.cphvb_com_free(ref m_component);
                    m_component.config = IntPtr.Zero;
                }
            }
        }
    }
}
