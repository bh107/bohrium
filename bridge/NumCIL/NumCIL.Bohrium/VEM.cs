//#define VIEW_BOOK_KEEPING

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
    /// Basic wrapper implementation of a Bohrium VEM
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
        /// ID for the user-defined random operation
        /// </summary>
        private readonly long m_randomFunctionId;

        /// <summary>
        /// ID for the user-defined maxtrix multiplication operation
        /// </summary>
        private readonly long m_matmulFunctionId;

        /// <summary>
        /// A reference to the Bohrium component for "self" aka the bridge
        /// </summary>
        private PInvoke.bh_component m_component;
        /// <summary>
        /// The unmanaged copy of the component
        /// </summary>
        private IntPtr m_componentPtr;
        /// <summary>
        /// A reference to the chpVB VEM
        /// </summary>
        private PInvoke.bh_component[] m_childs;
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
#if VIEW_BOOK_KEEPING
        /// <summary>
        /// A ref-counter for base arrays
        /// </summary>
        private Dictionary<PInvoke.bh_array_ptr, List<ViewPtrKeeper>> m_baseArrayRefs = new Dictionary<PInvoke.bh_array_ptr, List<ViewPtrKeeper>>();
#endif
        /// <summary>
        /// Lookup table for all created userfunc structures
        /// </summary>
        private Dictionary<IntPtr, GCHandle> m_allocatedUserfuncs = new Dictionary<IntPtr, GCHandle>();
        /// <summary>
        /// GC Handles for managed data
        /// </summary>
        private Dictionary<PInvoke.bh_array_ptr, GCHandle> m_managedHandles = new Dictionary<PInvoke.bh_array_ptr, GCHandle>();

		/// <summary>
		/// Gets a value indicating whether this instance is disposed.
		/// </summary>
		/// <value><c>true</c> if this instance is disposed; otherwise, <c>false</c>.</value>
		public bool IsDisposed { get; private set; }

        /// <summary>
        /// Constructs a new VEM
        /// </summary>
        public VEM()
        {
//Disable "Unreachable code" warning
#pragma warning disable 0162
            if (bh_opcode.BH_ADD == bh_opcode.BH_SUBTRACT)
                throw new Exception("This version of NumCIL.Bohrium contains invalid opcodes!");
#pragma warning restore

            m_component = PInvoke.bh_component_setup(out m_componentPtr);
            PInvoke.bh_error e = PInvoke.bh_component_children(m_component, out m_childs, out m_childsPtr);
            if (e != PInvoke.bh_error.BH_SUCCESS)
                throw new BohriumException(e);

            if (m_childs.Length > 1)
                throw new BohriumException(string.Format("Unexpected number of child nodes: {0}", m_childs.Length));

            e = m_childs[0].init(ref m_childs[0]);
            if (e != PInvoke.bh_error.BH_SUCCESS)
                throw new BohriumException(e);

            m_randomFunctionId = GetUserFuncId("bh_random");
            m_matmulFunctionId = GetUserFuncId("bh_matmul");
        }

        /// <summary>
        /// Helper to get the id of a userdefined function
        /// </summary>
        /// <param name="name">The name of the userfunc to find</param>
        /// <returns>The ID of the userdefined function or 0</returns>
        private long GetUserFuncId(string name)
        {
            //The exception only happens with the debugger attached
            long id = 0;
            try
            {
                PInvoke.bh_error e = m_childs[0].reg_func(name, ref id);
                if (e != PInvoke.bh_error.BH_SUCCESS)
                    id = 0;
            }
            catch { id = 0; }
            
            return id;
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
                    ExecuteRelease((PInvoke.bh_instruction)i);
        }

        /// <summary>
        /// Registers a release instruction for later execution, including a handle that must be disposed
        /// </summary>
        /// <param name="array">The array to discard</param>
        /// <param name="handle">The handle to dispose after discarding the array</param>
        public void ExecuteRelease(PInvoke.bh_array_ptr array, GCHandle handle)
        {
            lock (m_releaselock)
            {
                System.Diagnostics.Debug.Assert(array.BaseArray == PInvoke.bh_array_ptr.Null);
                if (handle.IsAllocated)
                    m_managedHandles.Add(array, handle);
                ExecuteRelease(new PInvoke.bh_instruction(bh_opcode.BH_DISCARD, array));
            }
        }




        /// <summary>
        /// Registers an instruction for later execution, usually destroy calls
        /// </summary>
        /// <param name="inst">The instruction to queue</param>
        public void ExecuteRelease(PInvoke.bh_instruction inst)
        {
            lock (m_releaselock)
            {
                if (inst.opcode == bh_opcode.BH_DISCARD)
                {
#if VIEW_BOOK_KEEPING
					var ar = inst.operand0;
                    if (ar.BaseArray != PInvoke.bh_array_ptr.Null)
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
#endif
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
		/// If the GC is responsible for collecting expired views and bases, 
		/// it returns these in random order, which is not valid in Bohrium.
		///
		///	This method will pick up the invalid order and swap instructions to
		/// ensure that the views are not discarded before the base array
		/// </summary>
		/// <returns>The input array</returns>
		/// <param name="instructions">The instructions to fix.</param>
		private static PInvoke.bh_instruction[] LowPassDiscardFixer(PInvoke.bh_instruction[] instructions)
		{
			var bases = new Dictionary<long, long>();
			for(var i = 0; i < instructions.Length; i++)
			{
				if (instructions[i].opcode == bh_opcode.BH_DISCARD && instructions[i].operand0 != PInvoke.bh_array_ptr.Null)
				{
					var pid = instructions[i].operand0.BaseArray == PInvoke.bh_array_ptr.Null ? instructions[i].operand0.PtrValue : instructions[i].operand0.BaseArray.PtrValue;
					if (instructions[i].operand0.BaseArray == PInvoke.bh_array_ptr.Null)
					{
						bases[pid] = i;
					}
					else
					{
						long ix;
						if (bases.TryGetValue(pid, out ix))
						{
							var n = instructions[ix];
							instructions[ix] = instructions[i];
							instructions[i] = n;
							bases[pid] = i;
						}
					}
					
				}
			}
			
			return instructions;
		}
		
		/// <summary>
        /// Executes a list of instructions
        /// </summary>
        /// <param name="inst_list">The list of instructions to execute</param>
        public void Execute(IEnumerable<IInstruction> inst_list)
        {
            var lst = inst_list;
            List<IInstruction> cleanup_lst = null;
            List<Tuple<long, PInvoke.bh_instruction, GCHandle>> handles = null;

            if (!m_preventCleanup && m_cleanups.Count > 0)
            {
                lock (m_releaselock)
                {
                    cleanup_lst = System.Threading.Interlocked.Exchange(ref m_cleanups, new List<IInstruction>());

                    GCHandle tmp;
                    long ix = inst_list.LongCount();
                    foreach (PInvoke.bh_instruction inst in cleanup_lst)
                    {
                        if (inst.opcode == bh_opcode.BH_DISCARD && m_managedHandles.TryGetValue(inst.operand0, out tmp))
                        {
                            if (handles == null)
                                handles = new List<Tuple<long, PInvoke.bh_instruction, GCHandle>>();
                            handles.Add(new Tuple<long, PInvoke.bh_instruction, GCHandle>(ix, inst, tmp));
                        }
                        ix++;
                    }

#if !VIEW_BOOK_KEEPING
					lst = lst.Concat(LowPassDiscardFixer(cleanup_lst.Cast<PInvoke.bh_instruction>().ToArray()).Cast<IInstruction>());
#else
					lst = lst.Concat(cleanup_lst);
#endif
                }
            }

            long errorIndex = -1;

            try
            {
                lock (m_executelock)
                    ExecuteWithoutLocks(lst, out errorIndex);
            }
            catch (Exception ex)
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
        /// Reshuffles instructions to honor Bohrium rules
        /// </summary>
        /// <param name="list">The list of instructions to reshuffle</param>
        private void ReshuffleInstructions(PInvoke.bh_instruction[] list)
        {
            if (list.LongLength <= 1)
                return;

            long lastIx = list.LongLength;
            for(long i = 0; i < lastIx; i++)
            {
                var inst = list[i];
                if (inst.opcode == bh_opcode.BH_DISCARD && inst.operand0.BaseArray == PInvoke.bh_array_ptr.Null)
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
        /// <param name="instList">The list of instructions to execute</param>
        /// <param name="errorIndex">A return value for the instruction that caused an error</param>
        private void ExecuteWithoutLocks(IEnumerable<IInstruction> instList, out long errorIndex)
        {
            var cleanups = new List<GCHandle>();
            long destroys = 0;
            errorIndex = -1;

            try
            {
                PInvoke.bh_instruction[] instrBuffer = instList.Select(x => (PInvoke.bh_instruction)x).ToArray();
                //ReshuffleInstructions(instrBuffer);

                foreach (var inst in instrBuffer)
                {
                    if (inst.opcode == bh_opcode.BH_DISCARD)
                        destroys++;
                    if (inst.userfunc != IntPtr.Zero)
                    {
                        cleanups.Add(m_allocatedUserfuncs[inst.userfunc]);
                        m_allocatedUserfuncs.Remove(inst.userfunc);
                    }
                }

                PInvoke.bh_error e = m_childs[0].execute(instrBuffer.LongLength, instrBuffer);

                if (e != PInvoke.bh_error.BH_SUCCESS)
                {
                    throw new BohriumException(e);
                }

                if (destroys > 0)
                    foreach (var inst in instrBuffer.Where(x => x.opcode == bh_opcode.BH_DISCARD))
                        PInvoke.bh_destroy_array(inst.operand0);
            }
            finally
            {
                foreach (var h in cleanups)
                    h.Free();
            }
        }

        /// <summary>
        /// Creates a Bohrium descriptor for a base array, without assigning the actual data
        /// </summary>
        /// <param name="d">The array to map</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.bh_array_ptr CreateBaseArray(Array d)
        {
            return CreateBaseArray(MapType(d.GetType().GetElementType()), d.LongLength);
        }

        /// <summary>
        /// Creates a base array with uninitialized memory
        /// </summary>
        /// <typeparam name="T">The data type for the array</typeparam>
        /// <param name="size">The size of the generated base array</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.bh_array_ptr CreateBaseArray<T>(long size)
        {
            return CreateBaseArray(MapType(typeof(T)), size);
        }

        /// <summary>
        /// Maps the element type to the Bohrium datatype
        /// </summary>
        /// <param name="t">The element type to look up</param>
        /// <returns>The Bohrium datatype</returns>
        public static PInvoke.bh_type MapType(Type t)
        {
            if (t == typeof(bool))
                return PInvoke.bh_type.BH_BOOL;
            else if (t == typeof(sbyte))
                return PInvoke.bh_type.BH_INT8;
            else if (t == typeof(short))
                return PInvoke.bh_type.BH_INT16;
            else if (t == typeof(int))
                return PInvoke.bh_type.BH_INT32;
            else if (t == typeof(long))
                return PInvoke.bh_type.BH_INT64;
            else if (t == typeof(byte))
                return PInvoke.bh_type.BH_UINT8;
            else if (t == typeof(ushort))
                return PInvoke.bh_type.BH_UINT16;
            else if (t == typeof(uint))
                return PInvoke.bh_type.BH_UINT32;
            else if (t == typeof(ulong))
                return PInvoke.bh_type.BH_UINT64;
            else if (t == typeof(float))
                return PInvoke.bh_type.BH_FLOAT32;
            else if (t == typeof(double))
                return PInvoke.bh_type.BH_FLOAT64;
            else if (t == typeof(NumCIL.Complex64.DataType))
                return PInvoke.bh_type.BH_COMPLEX64;
            else if (t == typeof(System.Numerics.Complex))
                return PInvoke.bh_type.BH_COMPLEX128;
            else
                throw new BohriumException(string.Format("Unsupported data type: " + t.FullName));
        }

        /// <summary>
        /// Creates a Bohrium base array or view descriptor
        /// </summary>
        /// <param name="type">The Bohrium type of data in the array</param>
        /// <param name="size">The size of the base array</param>
        /// <returns>The pointer to the base array descriptor</returns>
        public PInvoke.bh_array_ptr CreateBaseArray(PInvoke.bh_type type, long size)
        {
            var ptr = CreateArray(
                PInvoke.bh_array_ptr.Null,
                type,
                1,
                0,
                new long[] { size },
                new long[] { 1 }
            );

#if VIEW_BOOK_KEEPING
			lock (m_releaselock)
                m_baseArrayRefs.Add(ptr, new List<ViewPtrKeeper>());
#endif
            return ptr;
        }

        /// <summary>
        /// Creates a Bohrium view descriptor
        /// </summary>
        /// <param name="basearray">The base array pointer</param>
        /// <param name="type">The Bohrium type of data in the array</param>
        /// <param name="ndim">Number of dimensions</param>
        /// <param name="start">The offset into the base array</param>
        /// <param name="shape">The shape values for each dimension</param>
        /// <param name="stride">The stride values for each dimension</param>
        /// <returns>The pointer to the array descriptor</returns>
        public ViewPtrKeeper CreateView(PInvoke.bh_array_ptr basearray, PInvoke.bh_type type, long ndim, long start, long[] shape, long[] stride)
        {
            if (basearray == PInvoke.bh_array_ptr.Null)
                throw new ArgumentException("Base array cannot be null for a view");
            var ptr = new ViewPtrKeeper(CreateArray(basearray, type, ndim, start, shape, stride));
#if VIEW_BOOK_KEEPING
			lock (m_releaselock)
                m_baseArrayRefs[basearray].Add(ptr);
#endif
            return ptr;
        }

        /// <summary>
        /// Creates a Bohrium base array or view descriptor
        /// </summary>
        /// <param name="basearray">The base array pointer if creating a view or IntPtr.Zero if the view is a base array</param>
        /// <param name="type">The Bohrium type of data in the array</param>
        /// <param name="ndim">Number of dimensions</param>
        /// <param name="start">The offset into the base array</param>
        /// <param name="shape">The shape values for each dimension</param>
        /// <param name="stride">The stride values for each dimension</param>
        /// <returns>The pointer to the array descriptor</returns>
        protected PInvoke.bh_array_ptr CreateArray(PInvoke.bh_array_ptr basearray, PInvoke.bh_type type, long ndim, long start, long[] shape, long[] stride)
        {
            PInvoke.bh_error e;
            PInvoke.bh_array_ptr res;
            
            //TODO: We can just use an internal struct, and Pin it before calling
            lock (m_executelock)
                e = PInvoke.bh_create_array(basearray, type, ndim, start, shape, stride, out res);

            if (e != PInvoke.bh_error.BH_SUCCESS)
                throw new BohriumException(e);

            return res;
        }
		/// <summary>
		/// Releases all resources held
		/// </summary>
		public void Dispose()
		{
			Dispose(true);
		}
		
        /// <summary>
        /// Releases all resources held
        /// </summary>
        private void Dispose(bool disposing)
		{
			if (IsDisposed)
				return;
        	
			if (disposing)
				GC.SuppressFinalize(this);
        		
			//Ensure all views are collected
			GC.Collect();
			GC.WaitForPendingFinalizers();
            
#if VIEW_BOOK_KEEPING
			if (m_baseArrayRefs.Count > 0)
			{
				Console.WriteLine("WARNING: Found allocated views on shutdown");
				foreach (var k in m_baseArrayRefs.Values.ToArray())
					foreach (var m in k.ToArray())
						m.Dispose();

				GC.Collect();
				GC.WaitForPendingFinalizers();
			}

			if (m_baseArrayRefs.Count > 0)
				Console.WriteLine("WARNING: Some base arrays were stil allocated during VEM shutdown");
#endif

			m_preventCleanup = false;
			ExecuteCleanups();
			
			//From here on, we no longer accept new discards
			IsDisposed = true;
			
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
							PInvoke.bh_component_free(cur);
							cur += Marshal.SizeOf(typeof(IntPtr));
						}
						
						m_childsPtr = IntPtr.Zero;
					}
					
					m_childs = null;
				}
				
				if (m_componentPtr != IntPtr.Zero)
				{
					PInvoke.bh_component_free(m_componentPtr);
					m_component.config = IntPtr.Zero;
				}
				
				m_componentPtr = IntPtr.Zero;
			}
		}

        /// <summary>
        /// Finalizes the VEM and shuts down the Bohrium components
        /// </summary>
        ~VEM()
        {
            Dispose(false);
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
        protected ViewPtrKeeper CreateViewPtr<T>(PInvoke.bh_type type, NdArray<T> view)
        {
            PInvoke.bh_array_ptr basep;

            if (view.DataAccessor is BohriumAccessor<T>)
            {
                basep = ((BohriumAccessor<T>)view.DataAccessor).BaseArrayPtr;
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

            if (view.Tag as ViewPtrKeeper == null || ((ViewPtrKeeper)view.Tag).Pointer == PInvoke.bh_array_ptr.Null || ((ViewPtrKeeper)view.Tag).Pointer.BaseArray != basep)
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
        public ViewPtrKeeper CreateView<T>(Shape shape, PInvoke.bh_array_ptr baseArray)
        {
            return CreateView(MapType(typeof(T)), shape, baseArray);
        }

        /// <summary>
        /// Creates a new view of data
        /// </summary>
        /// <param name="BH_TYPE">The data type of the view</param>
        /// <param name="shape">The shape to create the view for</param>
        /// <param name="baseArray">The array to set as base array</param>
        /// <returns>A new view</returns>
        public ViewPtrKeeper CreateView(PInvoke.bh_type BH_TYPE, Shape shape, PInvoke.bh_array_ptr baseArray)
        {
            //Unroll, to avoid creating a Linq query for basic 3d shapes
            switch(shape.Dimensions.Length)
            {
                case 1:
                    return CreateView(
                        baseArray,
                        BH_TYPE,
                        shape.Dimensions.Length,
                        (int)shape.Offset,
                        new long[] { shape.Dimensions[0].Length },
                        new long[] { shape.Dimensions[0].Stride }
                    );
                case 2:
                    return CreateView(
                            baseArray,
                            BH_TYPE,
                            shape.Dimensions.Length,
                            (int)shape.Offset,
                            new long[] { shape.Dimensions[0].Length, shape.Dimensions[1].Length },
                            new long[] { shape.Dimensions[0].Stride, shape.Dimensions[1].Stride }
                        );
                case 3:
                    return CreateView(
                        baseArray,
                        BH_TYPE,
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
                        BH_TYPE,
                        shape.Dimensions.Length,
                        (int)shape.Offset,
                        lengths,
                        strides
                    );
            }
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="operand">The output operand</param>
        /// <param name="constant">An optional constant value</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(bh_opcode opcode, NdArray<T> operand, PInvoke.bh_constant constant = new PInvoke.bh_constant())
        {
            return CreateInstruction<T>(MapType(typeof(T)), opcode, operand);
        }
        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="op1">The output operand</param>
        /// <param name="op2">The input operand</param>
        /// <param name="constant">An optional constant value</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(bh_opcode opcode, NdArray<T> op1, NdArray<T> op2, PInvoke.bh_constant constant = new PInvoke.bh_constant())
        {
            return CreateInstruction<T>(MapType(typeof(T)), opcode, op1, op2);
        }
        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="op1">The output operand</param>
        /// <param name="op2">An input operand</param>
        /// <param name="op3">Another input operand</param>
        /// <param name="constant">An optional constant value</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(bh_opcode opcode, NdArray<T> op1, NdArray<T> op2, NdArray<T> op3, PInvoke.bh_constant constant = new PInvoke.bh_constant())
        {
            return CreateInstruction<T>(MapType(typeof(T)), opcode, op1, op2, op3);
        }
        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="operands">A list of operands</param>
        /// <param name="constant">An optional constant value</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(bh_opcode opcode, IEnumerable<NdArray<T>> operands, PInvoke.bh_constant constant = new PInvoke.bh_constant())
        {
            return CreateInstruction<T>(MapType(typeof(T)), opcode, operands);
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="operand">The output operand</param>
        /// <param name="constant">An optional constant value</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(PInvoke.bh_type type, bh_opcode opcode, NdArray<T> operand, PInvoke.bh_constant constant = new PInvoke.bh_constant())
        {
            return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, operand).Pointer, constant);
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="op1">The output operand</param>
        /// <param name="op2">The input operand</param>
        /// <param name="constant">An optional constant value</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(PInvoke.bh_type type, bh_opcode opcode, NdArray<T> op1, NdArray<T> op2, PInvoke.bh_constant constant = new PInvoke.bh_constant())
        {
            return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, CreateViewPtr<T>(type, op2).Pointer, constant);
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="op1">The output operand</param>
        /// <param name="op2">The input operand</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(PInvoke.bh_type type, bh_opcode opcode, NdArray<T> op1, NdArray<T> op2)
        {
            if (IsScalar(op2))
                return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, PInvoke.bh_array_ptr.Null, new PInvoke.bh_constant(type, op2.DataAccessor[0]));
            else
                return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, CreateViewPtr<T>(type, op2).Pointer, new PInvoke.bh_constant());
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="op1">The output operand</param>
        /// <param name="constant">An left-hand-side constant value</param>
        /// <param name="op2">The input operand</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(PInvoke.bh_type type, bh_opcode opcode, NdArray<T> op1, PInvoke.bh_constant constant, NdArray<T> op2)
        {
            return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, constant, CreateViewPtr<T>(type, op2).Pointer);
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="op1">The output operand</param>
        /// <param name="op2">An input operand</param>
        /// <param name="op3">Another input operand</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(PInvoke.bh_type type, bh_opcode opcode, NdArray<T> op1, NdArray<T> op2, NdArray<T> op3)
        {
            if (IsScalar(op2))
                return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, PInvoke.bh_array_ptr.Null, CreateViewPtr<T>(type, op3).Pointer, new PInvoke.bh_constant(type, op2.DataAccessor[0]));
            else if (IsScalar(op3))
                return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, CreateViewPtr<T>(type, op2).Pointer, PInvoke.bh_array_ptr.Null, new PInvoke.bh_constant(type, op3.DataAccessor[0]));
            else
                return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, CreateViewPtr<T>(type, op2).Pointer, CreateViewPtr<T>(type, op3).Pointer, new PInvoke.bh_constant());
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="op1">The output operand</param>
        /// <param name="op2">An input operand</param>
        /// <param name="op3">Another input operand</param>
        /// <param name="constant">A constant value</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(PInvoke.bh_type type, bh_opcode opcode, NdArray<T> op1, NdArray<T> op2, NdArray<T> op3, PInvoke.bh_constant constant)
        {
            return new PInvoke.bh_instruction(opcode, CreateViewPtr<T>(type, op1).Pointer, CreateViewPtr<T>(type, op2).Pointer, CreateViewPtr<T>(type, op3).Pointer, constant);
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="operands">A list of operands</param>
        /// <param name="constant">A constant value</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(PInvoke.bh_type type, bh_opcode opcode, IEnumerable<NdArray<T>> operands, PInvoke.bh_constant constant)
        {
            return new PInvoke.bh_instruction(opcode, operands.Select(x => CreateViewPtr<T>(type, x).Pointer), constant);
        }

        /// <summary>
        /// Creates a new instruction
        /// </summary>
        /// <typeparam name="T">The type of data used in the instruction</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="operands">A list of operands</param>
        /// <returns>The new instruction</returns>
        public IInstruction CreateInstruction<T>(PInvoke.bh_type type, bh_opcode opcode, IEnumerable<NdArray<T>> operands)
        {
            bool constantUsed = false;
            PInvoke.bh_constant constant = new PInvoke.bh_constant();

            return new PInvoke.bh_instruction(opcode, operands.Select(x => {
                if (!constantUsed && IsScalar(x))
                {
                    constant = new PInvoke.bh_constant(type, x.DataAccessor[0]);
                    return PInvoke.bh_array_ptr.Null;
                }
                else
                    return CreateViewPtr<T>(type, x).Pointer;
            }), constant);
        }

        /// <summary>
        /// Creates a new instruction that convers from Tb to Ta
        /// </summary>
        /// <typeparam name="Ta">The output element datatype</typeparam>
        /// <typeparam name="Tb">The input element datatype</typeparam>
        /// <param name="supported">A list of accumulated instructions</param>
        /// <param name="opcode">The instruction opcode</param>
        /// <param name="typea">The Bohrium datatype for the output</param>
        /// <param name="output">The output operand</param>
        /// <param name="in1">An input operand</param>
        /// <param name="in2">Another input operand</param>
        /// <returns>A new instruction</returns>
        public IInstruction CreateConversionInstruction<Ta, Tb>(List<IInstruction> supported, NumCIL.Bohrium.bh_opcode opcode, PInvoke.bh_type typea, NdArray<Ta> output, NdArray<Tb> in1, NdArray<Tb> in2)
        {
            if (IsScalar(in1))
                return new PInvoke.bh_instruction(opcode, CreateViewPtr<Ta>(typea, output).Pointer, new PInvoke.bh_constant(in1.DataAccessor[0]), in2 == null ? PInvoke.bh_array_ptr.Null : CreateViewPtr<Tb>(in2).Pointer);
            else if (in2 != null && IsScalar(in2))
                return new PInvoke.bh_instruction(opcode, CreateViewPtr<Ta>(typea, output).Pointer, CreateViewPtr<Tb>(in1).Pointer, new PInvoke.bh_constant(in2.DataAccessor[0]));
            else
                return new PInvoke.bh_instruction(opcode, CreateViewPtr<Ta>(typea, output).Pointer, CreateViewPtr<Tb>(in1).Pointer, in2 == null ? PInvoke.bh_array_ptr.Null : CreateViewPtr<Tb>(in2).Pointer);
        }

        /// <summary>
        /// Creates a new random userfunc instruction
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="op1">The output operand</param>
        /// <returns>A new instruction</returns>
        public IInstruction CreateRandomInstruction<T>(PInvoke.bh_type type, NdArray<T> op1)
        {
            if (!SupportsRandom)
                throw new BohriumException("The VEM/VE setup does not support the random function");

            if (op1.Shape.Offset != 0 || !op1.Shape.IsPlain || op1.Shape.Elements != op1.DataAccessor.Length)
                throw new Exception("The shape of the element that is sent to the random implementation must be a non-shape plain array");

            GCHandle gh = GCHandle.Alloc(
                new PInvoke.bh_userfunc_random(
                    m_randomFunctionId,
                    CreateViewPtr<T>(type, op1).Pointer.BaseArray
                ), 
                GCHandleType.Pinned
            );

            IntPtr adr = gh.AddrOfPinnedObject();

            m_allocatedUserfuncs.Add(adr, gh);

            return new PInvoke.bh_instruction(
                bh_opcode.BH_USERFUNC,
                adr                    
            );
        }

        /// <summary>
        /// Creats a new matmul userfunc
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <param name="type">The Bohrium datatype</param>
        /// <param name="op1">The output operand</param>
        /// <param name="op2">An input operand</param>
        /// <param name="op3">Another input operand</param>
        /// <returns>A new instruction</returns>
        public IInstruction CreateMatmulInstruction<T>(PInvoke.bh_type type, NdArray<T> op1, NdArray<T> op2, NdArray<T> op3)
        {
            if (!SupportsMatmul)
                throw new BohriumException("The VEM/VE setup does not support the matmul function");

            GCHandle gh = GCHandle.Alloc(
                new PInvoke.bh_userfunc_matmul(
                    m_matmulFunctionId,
                    CreateViewPtr<T>(type, op1).Pointer,
                    CreateViewPtr<T>(type, op2).Pointer,
                    CreateViewPtr<T>(type, op3).Pointer
                ),
                GCHandleType.Pinned
            );

            IntPtr adr = gh.AddrOfPinnedObject();

            m_allocatedUserfuncs.Add(adr, gh);

            return new PInvoke.bh_instruction(
                bh_opcode.BH_USERFUNC,
                adr
            );
        }

        /// <summary>
        /// Returns a value indicating if a value is a scalar
        /// </summary>
        /// <typeparam name="T">The type of data in the array</typeparam>
        /// <param name="ar">The array to examine</param>
        /// <returns>True if the alue can be represented as a Bohrium constant, false otherwise</returns>
        private static bool IsScalar<T>(NdArray<T> ar)
        {
            if (ar.DataAccessor.Length == 1)
                if (ar.DataAccessor.GetType() == typeof(DefaultAccessor<T>))
                    return true;
                else if (ar.DataAccessor.GetType() == typeof(BohriumAccessor<T>) && ar.DataAccessor.IsAllocated)
                    return true;

            return false;
        }
        
        /// <summary>
        /// Determines whether the instruction has valid types
        /// </summary>
        /// <returns><c>true</c> if the instruction has valid types otherwise, <c>false</c>.</returns>
        /// <param name="inst">The instruction to validate</param>
        public bool IsValidInstruction(IInstruction inst)
		{
			var i = (PInvoke.bh_instruction)inst;
			var out_type = i.operand0.Type;
			var nops = PInvoke.bh_operands(inst.OpCode);
			if (nops == 1)
				return PInvoke.bh_validate_types(inst.OpCode, out_type, PInvoke.bh_type.BH_UNKNOWN, PInvoke.bh_type.BH_UNKNOWN, PInvoke.bh_type.BH_UNKNOWN);
			
			var input1 = i.operand1 == PInvoke.bh_array_ptr.Null ? PInvoke.bh_type.BH_UNKNOWN : i.operand1.Type;
			var input2 = i.operand2 == PInvoke.bh_array_ptr.Null ? PInvoke.bh_type.BH_UNKNOWN : i.operand2.Type;
			var scalar = i.constant.type;
			
			if (nops == 2 || nops == 3)
				return PInvoke.bh_validate_types(inst.OpCode, out_type, input1, input2, scalar);

			throw new Exception(string.Format("Unexpected number of operands for opcode {1}: {0}", inst.OpCode, nops));
        }
        
        /*public IInstruction[] GetConversionSequence(IInstruction inst)
		{
			var i = (PInvoke.bh_instruction)inst;
			var out_type = i.operand0.Type;
			var nops = PInvoke.bh_operands(inst.OpCode);
			if (nops == 1)
				return null;
			
			var input1 = i.operand1 == PInvoke.bh_array_ptr.Null ? PInvoke.bh_type.BH_UNKNOWN : i.operand1.Type;
			var input2 = i.operand2 == PInvoke.bh_array_ptr.Null ? PInvoke.bh_type.BH_UNKNOWN : i.operand2.Type;
			var scalar = i.constant.type;
			
			if (nops == 2 || nops == 3)
			{
				var orig1 = input1;
				var orig2 = input2;
				var origs = scalar;
				
				if(!PInvoke.bh_get_type_conversion(inst.OpCode, out_type, ref input1, ref input2, ref scalar))
					return null;
					
				var conversions = new List<IInstruction>();
				if (input1 != orig1)
					conversions.Add(CreateConversionInstruction);
				if (input2 != orig2)
					conversions.Add(CreateConversionInstruction);
				if (origs != scalar)
					throw new Exception("Scalar conversion requested to match types, this is an error in the library");
			}
			
			throw new Exception(string.Format("Unexpected number of operands for opcode {1}: {0}", inst.OpCode, nops));
		}*/

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
