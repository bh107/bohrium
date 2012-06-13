using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.Generic
{
    /// <summary>
    /// Simple interface that describes a basic flat array
    /// </summary>
    /// <typeparam name="T">The type of data in the array</typeparam>
    public interface IDataAccessor<T>
    {
        /// <summary>
        /// Gets the number of elements in the array
        /// </summary>
        long Length { get; }

        /// <summary>
        /// Gets the data as a .Net Array
        /// </summary>
        T[] AsArray();

        /// <summary>
        /// Ensures that data is allocated
        /// </summary>
        void Allocate();

        /// <summary>
        /// Returns a value indicating if the array is allocated
        /// </summary>
        bool IsAllocated { get; }

        /// <summary>
        /// Gets or sets the value at a specific index.
        /// Depending on implementation, this may cause the array to be allocated.
        /// </summary>
        /// <param name="index">The index to get or set the value at</param>
        /// <returns>The value at the given index</returns>
        T this[long index] { get; set; }

        /// <summary>
        /// An extra component that can be used to tag data to the accessor
        /// </summary>
        object Tag { get; set; }
    }

    /// <summary>
    /// Interface to data that is not kept in managed memory
    /// </summary>
    /// <typeparam name="T">The type of data in the array</typeparam>
    public interface IUnmanagedDataAccessor<T> : IDataAccessor<T>
    {
        /// <summary>
        /// Gets a pointer to the data
        /// </summary>
        IntPtr Pointer { get; }

        /// <summary>
        /// Gets a value indicating if it is possible to return the data as a .Net array
        /// </summary>
        bool CanAllocateArray { get; }
    }

    /// <summary>
    /// Interface that adds a lazy registration function to a data accessor
    /// </summary>
    /// <typeparam name="T">The type of data in the array</typeparam>
    public interface ILazyAccessor<T> : IDataAccessor<T>
    {
        /// <summary>
        /// Register a pending operation on the underlying array
        /// </summary>
        /// <param name="operation">The operation performed</param>
        /// <param name="operands">The operands involved, operand 0 is the target</param>
        void AddOperation(IOp<T> operation, params NdArray<T>[] operands);

        /// <summary>
        /// Gets a list of registered pending operations on the accessor
        /// </summary>
        IList<PendingOperation<T>> PendingOperations { get; }

        /// <summary>
        /// The number of already executed operations
        /// </summary>
        long PendignOperationOffset { get; set; }

        /// <summary>
        /// Flushes all pending operations on this element
        /// </summary>
        void Flush();
    }

    /// <summary>
    /// Interface for creating accessors
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public interface IAccessorFactory<T>
    {
        /// <summary>
        /// Creates a new data accessor for an array of the given size
        /// </summary>
        /// <param name="size">The size of the array to create an accessor for</param>
        /// <returns>A new accessor</returns>
        IDataAccessor<T> Create(long size);

        /// <summary>
        /// Creates a new data accessor for an allocated array
        /// </summary>
        /// <param name="data">The array to create an accessor for</param>
        /// <returns>A new accessor</returns>
        IDataAccessor<T> Create(T[] data);
    }

    /// <summary>
    /// Wrapper implementation for a normal .Net array,
    /// which is allocated when first accessed
    /// </summary>
    /// <typeparam name="T">The type of data in the array</typeparam>
    public class DefaultAccessor<T> : IDataAccessor<T>
    {
        /// <summary>
        /// The actual data storage
        /// </summary>
        protected T[] m_data = null;

        /// <summary>
        /// The size of the data
        /// </summary>
        protected long m_size;

        /// <summary>
        /// An accessor tag
        /// </summary>
        public object Tag { get; set; }

        /// <summary>
        /// Constructs a wrapper around an existing arrray
        /// </summary>
        /// <param name="data">The data the accessor represents</param>
        public DefaultAccessor(T[] data)
        {
            if (data == null)
                throw new ArgumentNullException("data");

            m_size = data.LongLength;
            m_data = data;
        }
        
        /// <summary>
        /// Constructs a placeholder for an array of a certain size
        /// </summary>
        /// <param name="size">The number of elements in the array</param>
        public DefaultAccessor(long size)
        {
            if (size < 0)
                throw new ArgumentOutOfRangeException("size");

            m_size = size;
        }

        /// <summary>
        /// Allocates data
        /// </summary>
        public virtual void Allocate()
        {
            if (m_data == null)
                m_data = new T[m_size];
        }

        /// <summary>
        /// Returns the value at a given index, this will allocated the array
        /// </summary>
        /// <param name="index">The index to get the value for</param>
        /// <returns>The value at the given index</returns>
        public virtual T this[long index]
        {
            get { Allocate(); return m_data[index]; }
            set { Allocate(); m_data[index] = value; }
        }

        /// <summary>
        /// Allocates data and returns the array
        /// </summary>
        /// <returns>The allocated data block</returns>
        public virtual T[] AsArray()
        {
            Allocate();
            return m_data;
        }

        /// <summary>
        /// Gets the size of the array
        /// </summary>
        public virtual long Length { get { return m_size; } }

        /// <summary>
        /// Gets a value indicating if the data is allocated
        /// </summary>
        public virtual bool IsAllocated { get { return m_data != null; } }
    }

    /// <summary>
    /// Implementation of a lazy initialized array, will collect operations until data is accessed
    /// </summary>
    /// <typeparam name="T">The type of data in the array</typeparam>
    public class LazyAccessor<T> : DefaultAccessor<T>, ILazyAccessor<T>
    {
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo binaryBaseMethodType = typeof(UFunc.FlushMethods).GetMethod("ApplyBinaryOp", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo unaryBaseMethodType = typeof(UFunc.FlushMethods).GetMethod("ApplyUnaryOp", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo nullaryBaseMethodType = typeof(UFunc.FlushMethods).GetMethod("ApplyNullaryOp", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo reduceBaseMethodType = typeof(UFunc.FlushMethods).GetMethod("Reduce", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo matmulBaseMethodType = typeof(UFunc.FlushMethods).GetMethod("Matmul", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of instantiated template methods
        /// </summary>
        protected static readonly Dictionary<object, System.Reflection.MethodInfo> specializedMethods = new Dictionary<object, System.Reflection.MethodInfo>();

        /// <summary>
        /// List of operations registered on this array but not yet executed
        /// </summary>
        protected List<PendingOperation<T>> m_pendingOperations = new List<PendingOperation<T>>();
        /// <summary>
        /// Offset used to calculate index after cleaning the pending operations
        /// </summary>
        protected long m_pendingOperationOffset;

        /// <summary>
        /// Locking object to allow nice threading properties
        /// </summary>
        public readonly object Lock = new object();


        /// <summary>
        /// Constructs a wrapper around an existing arrray
        /// </summary>
        /// <param name="data">The data the accessor represents</param>
        public LazyAccessor(T[] data) : base(data) { }
        /// <summary>
        /// Constructs a placeholder for an array of a certain size
        /// </summary>
        /// <param name="size">The number of elements in the array</param>
        public LazyAccessor(long size) : base(size) { }

        /// <summary>
        /// Allocates that data, calling this method will allocate memory,
        /// and execute all pending operations
        /// </summary>
        public override void Allocate()
        {
            if (PendingOperations.Count != 0)
                this.ExecutePendingOperations();

            base.Allocate();
        }

        /// <summary>
        /// Register a pending operation on the underlying array
        /// </summary>
        /// <param name="operation">The operation performed</param>
        /// <param name="operands">The operands involved, operand 0 is the target</param>
        public virtual void AddOperation(IOp<T> operation, params NdArray<T>[] operands)
        {
            lock (Lock)
                PendingOperations.Add(new PendingOperation<T>(operation, operands));
        }

        /// <summary>
        /// Execute all operations that are pending to obtain the result array
        /// </summary>
        protected virtual void ExecutePendingOperations()
        {
            if (PendingOperations.Count > 0)
            {
                var lst = UnrollWorkList(this);
                ExecuteOperations(lst);
                PendingOperations.Clear();
            }
        }

        /// <summary>
        /// Flushes all instructions queued on this element
        /// </summary>
        public void Flush()
        {
            ExecutePendingOperations();
        }

        /// <summary>
        /// Function that builds a serialized list of operations to execute to obtain the target output
        /// </summary>
        /// <param name="target">The target output</param>
        /// <returns>A list of operations to perform</returns>
        public virtual IEnumerable<PendingOperation<T>> UnrollWorkList(ILazyAccessor<T> target)
        {
            List<PendingOperation<T>> res = new List<PendingOperation<T>>();
            Dictionary<ILazyAccessor<T>, long> completedOps = new Dictionary<ILazyAccessor<T>, long>();
            res.AddRange(target.PendingOperations);
            completedOps[target] = target.PendingOperations.Count + target.PendignOperationOffset;

            //Figure out which operations we need
            long i = 0;
            while (i < res.Count)
            {
                PendingOperation<T> cur = res[(int)i];

                for (int j = 0; j < cur.Operands.Length; j++)
                    if (cur.Operands[j].DataAccessor is ILazyAccessor<T>)
                    {
                        ILazyAccessor<T> lz = (ILazyAccessor<T>)cur.Operands[j].DataAccessor;
                        long cp;
                        long dest_cp = cur.OperandIndex[j] + (j == 0 ? -1 : 0);

                        if (!completedOps.TryGetValue(lz, out cp))
                            cp = lz.PendignOperationOffset;

                        long max_cp = Math.Max(cp, dest_cp);
                        for (long k = cp; k < max_cp; k++)
                            res.Add(lz.PendingOperations[(int)(k - cp)]);

                        completedOps[lz] = max_cp;
                    }

                i++;
            }

            //Now we collect the operations that we need to execute and mark them as executed in the accessor
            foreach (var kp in completedOps)
            {
                long oldOffset = kp.Key.PendignOperationOffset;

                if (kp.Value - oldOffset == 0)
                    kp.Key.PendingOperations.Clear();
                else
                    for (i = oldOffset; i < kp.Value; i++)
                        kp.Key.PendingOperations.RemoveAt(0);
                
                kp.Key.PendignOperationOffset = kp.Value;
            }

            //Sort list by clock
            IEnumerable<PendingOperation<T>> tmp = res.OrderBy(x => x.Clock);

            //Remove duplicates
            /*long prevclock = -1;
            tmp = tmp.Where((x) =>
            {
                if (x.Clock == prevclock)
                    return false;

                prevclock = x.Clock;
                return true;
            });*/

            return tmp;
        }

        /// <summary>
        /// Basic execution function, simply calls the UFunc*Flush functions with the pending operation
        /// </summary>
        /// <param name="work">The list of operations to perform</param>
        public virtual void ExecuteOperations(IEnumerable<PendingOperation<T>> work)
        {
            foreach (var n in work)
            {
                if (n.Operation is NumCIL.UFunc.LazyReduceOperation<T>)
                {
                    NumCIL.UFunc.LazyReduceOperation<T> lzop = (NumCIL.UFunc.LazyReduceOperation<T>)n.Operation;

                    System.Reflection.MethodInfo genericVersion;
                    if (!specializedMethods.TryGetValue(lzop.Operation.GetType(), out genericVersion))
                    {
                        genericVersion = reduceBaseMethodType.MakeGenericMethod(typeof(T), lzop.Operation.GetType());
                        specializedMethods[lzop.Operation.GetType()] = genericVersion;
                    }

                    genericVersion.Invoke(null, new object[] { lzop.Operation, lzop.Axis, n.Operands[1], n.Operands[0] });
                    
                }
                else if (n.Operation is NumCIL.UFunc.LazyMatmulOperation<T>)
                {
                    NumCIL.UFunc.LazyMatmulOperation<T> lzmt = (NumCIL.UFunc.LazyMatmulOperation<T>)n.Operation;

                    System.Reflection.MethodInfo genericVersion = matmulBaseMethodType.MakeGenericMethod(typeof(T), lzmt.AddOperator.GetType(), lzmt.MulOperator.GetType());
                    genericVersion.Invoke(null, new object[] { lzmt.AddOperator, lzmt.MulOperator, n.Operands[1], n.Operands[2], n.Operands[0] });

                }
                else if (n.Operation is IBinaryOp<T>)
                {
                    System.Reflection.MethodInfo genericVersion;
                    if (!specializedMethods.TryGetValue(n.Operation, out genericVersion))
                    {
                        genericVersion = binaryBaseMethodType.MakeGenericMethod(typeof(T), n.Operation.GetType());
                        specializedMethods[n.Operation] = genericVersion;
                    }

                    genericVersion.Invoke(null, new object[] { n.Operation, n.Operands[1], n.Operands[2], n.Operands[0] });
                }
                else if (n.Operation is IUnaryOp<T>)
                {
                    System.Reflection.MethodInfo genericVersion;
                    if (!specializedMethods.TryGetValue(n.Operation, out genericVersion))
                    {
                        genericVersion = unaryBaseMethodType.MakeGenericMethod(typeof(T), n.Operation.GetType());
                        specializedMethods[n.Operation] = genericVersion;
                    }
                    genericVersion.Invoke(null, new object[] { n.Operation, n.Operands[1], n.Operands[0] });
                }
                else if (n.Operation is INullaryOp<T>)
                {
                    System.Reflection.MethodInfo genericVersion;
                    if (!specializedMethods.TryGetValue(n.Operation, out genericVersion))
                    {
                        genericVersion = nullaryBaseMethodType.MakeGenericMethod(typeof(T), n.Operation.GetType());
                        specializedMethods[n.Operation] = genericVersion;
                    }
                    genericVersion.Invoke(null, new object[] { n.Operation, n.Operands[0] });
                }
                else
                {
                    throw new Exception("Unexpected operation");
                }
            }
        }


        /// <summary>
        /// Gets a list of registered pending operations on the accessor
        /// </summary>
        public IList<PendingOperation<T>> PendingOperations
        {
            get { return m_pendingOperations; }
        }

        /// <summary>
        /// The number of already executed operations
        /// </summary>
        public long PendignOperationOffset
        {
            get { return m_pendingOperationOffset; }
            set { m_pendingOperationOffset = value; }
        }
    }


    /// <summary>
    /// Representation of a pending operation
    /// </summary>
    /// <typeparam name="T">The type of data in the array</typeparam>
    public class PendingOperation<T>
    {
        /// <summary>
        /// We keep a global clock on all operations so we can easily sort them later
        /// </summary>
        protected static long _globalClock = 0;

        /// <summary>
        /// The relative time this operation was registered
        /// </summary>
        public readonly long Clock;
        /// <summary>
        /// The operation to perform, usually a IBinaryOp&lt;T&gt; or IUnaryOp&lt;T&gt;
        /// </summary>
        public readonly IOp<T> Operation;

        /// <summary>
        /// The list of operands involved in this operation,
        /// the target operand is at index 0
        /// </summary>
        public readonly NdArray<T>[] Operands;
        /// <summary>
        /// The size of pending operations after the execution,
        /// for each of the operands
        /// </summary>
        public readonly long[] OperandIndex;

        /// <summary>
        /// Constructs a new pending operation
        /// </summary>
        /// <param name="operation">The operation to perform</param>
        /// <param name="operands">The operands involved</param>
        public PendingOperation(IOp<T> operation, params NdArray<T>[] operands)
        {
            this.Clock = System.Threading.Interlocked.Increment(ref _globalClock);
            this.Operation = operation;

            NdArray<T>[] oprs = new NdArray<T>[operands.Length];
            long[] indx = new long[operands.Length];
            int i = 0;

            foreach (var x in operands)
            {
                oprs[i] = x;
                if (x.DataAccessor is ILazyAccessor<T>)
                {
                    ILazyAccessor<T> lz = (ILazyAccessor<T>)x.DataAccessor;
                    indx[i] = (lz.PendingOperations.Count + lz.PendignOperationOffset) + (i == 0 ? 1 : 0);
                }
                else
                    indx[i] = i == 0 ? 1 : 0;
                i++;
            }

            this.Operands = operands;
            this.OperandIndex = indx;
        }
    }

    /// <summary>
    /// Default factory for creating data accessors
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public class DefaultAccessorFactory<T> : IAccessorFactory<T>
    {
        /// <summary>
        /// The size of the elements generated by this factory
        /// </summary>
        protected static readonly long NATIVE_ELEMENT_SIZE = System.Runtime.InteropServices.Marshal.SizeOf(typeof(T));

        /// <summary>
        /// Creates a new data accessor for an array of the given size
        /// </summary>
        /// <param name="size">The size of the array to create an accessor for</param>
        /// <returns>A new accessor</returns>
        public IDataAccessor<T> Create(long size) 
        { 
            IDataAccessor<T> result = null;

            if (UnsafeAPI.IsUnsafeSupported && !UnsafeAPI.DisableUnsafeAPI && !UnsafeAPI.DisableUnsafeArrays && (size * NATIVE_ELEMENT_SIZE) >= UnsafeAPI.UnsafeArraysLargerThan)
                result = UnsafeAPI.CreateAccessor<T>(size);

            return result ?? new DefaultAccessor<T>(size); 
        }
        /// <summary>
        /// Creates a new data accessor for an allocated array
        /// </summary>
        /// <param name="data">The array to create an accessor for</param>
        /// <returns>A new accessor</returns>
        public IDataAccessor<T> Create(T[] data) 
        {
            return new DefaultAccessor<T>(data); 
        }
    }

    /// <summary>
    /// Default factory for creating data accessors
    /// </summary>
    /// <typeparam name="T">The type of data to operate on</typeparam>
    public class LazyAccessorFactory<T> : IAccessorFactory<T>
    {
        /// <summary>
        /// Creates a new data accessor for an array of the given size
        /// </summary>
        /// <param name="size">The size of the array to create an accessor for</param>
        /// <returns>A new accessor</returns>
        public IDataAccessor<T> Create(long size) { return new LazyAccessor<T>(size); }
        /// <summary>
        /// Creates a new data accessor for an allocated array
        /// </summary>
        /// <param name="data">The array to create an accessor for</param>
        /// <returns>A new accessor</returns>
        public IDataAccessor<T> Create(T[] data) { return new LazyAccessor<T>(data); }
    }

}
