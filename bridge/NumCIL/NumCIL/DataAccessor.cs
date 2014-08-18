#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium:
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
    /// Interface for marking an accessor flush capable
    /// </summary>
    public interface IFlushableAccessor
    {
        /// <summary>
        /// Flushes all pending operations on this element
        /// </summary>
        void Flush();
    }

    /// <summary>
    /// Interface that adds a lazy registration function to a data accessor
    /// </summary>
    /// <typeparam name="T">The type of data in the array</typeparam>
    public interface ILazyAccessor<T> : IDataAccessor<T>, IFlushableAccessor
    {
        /// <summary>
        /// Register a pending operation on the underlying array
        /// </summary>
        /// <param name="operation">The operation performed</param>
        /// <param name="operands">The operands involved, operand 0 is the target</param>
        void AddOperation(IOp<T> operation, params NdArray<T>[] operands);

		/// <summary>
        /// Register a pending operation on the underlying array
        /// </summary>
        /// <param name="operation">The operation performed</param>
        /// <param name="output">The output operand</param>
        /// <param name="input">The input operand</param>
        /// <typeparam name="Tb">The source data type</typeparam>
        void AddConversionOperation<Tb>(IUnaryConvOp<Tb, T> operation, NdArray<T> output, NdArray<Tb> input);

        /// <summary>
        /// Register a pending operation on the underlying array
        /// </summary>
        /// <param name="operation">The operation performed</param>
        /// <param name="output">The output operand</param>
        /// <param name="in1">An input operand</param>
        /// <param name="in2">An input operand</param>
        /// <typeparam name="Tb">The source data type</typeparam>
        void AddConversionOperation<Tb>(IBinaryConvOp<Tb, T> operation, NdArray<T> output, NdArray<Tb> in1, NdArray<Tb> in2);
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
    /// The collection point for lazily evaluated expressions
    /// </summary>
	public static class LazyAccessorCollector
	{
		/// <summary>
		/// We keep a global clock on all operations so we can easily sort them later
		/// </summary>
		private static long _globalClock = 0;

		/// <summary>
		/// List of operations registered on this array but not yet executed
		/// </summary>
		private static List<IPendingOperation> _pendingOperations = new List<IPendingOperation>();
		
		/// <summary>
		/// The lock guarding the pending operations
		/// </summary>
		private static readonly object _pendingOperationsLock = new object();

		/// <summary>
		/// Adds an operation to the list of pending operations
		/// </summary>
		/// <returns>The operation clock</returns>
		/// <param name="op">The operation to add</param>
		public static long AddOperation(IPendingOperation op)
		{
			lock(_pendingOperationsLock)
				_pendingOperations.Add(op);
			
			return op.Clock;
		}
		
		/// <summary>
		/// Gets the clock tick and increments it atomically.
		/// </summary>
		/// <value>The clock tick.</value>
		public static long ClockTick { get { return System.Threading.Interlocked.Increment(ref _globalClock); } }

		/// <summary>
		/// Extracts all operations with a clock less than or equal to the specified clock
		/// </summary>
		/// <returns>All operations with a clock less than or equal to the specified max</returns>
		/// <param name="maxclock">The maximum clock</param>
		public static IList<IPendingOperation> ExtractUntilClock(long maxclock)
		{
			lock (_pendingOperationsLock)
			{
				var tmp = new List<IPendingOperation>();
				while (_pendingOperations.Count > 0 && _pendingOperations[0].Clock <= maxclock)
				{
					tmp.Add(_pendingOperations[0]);
					_pendingOperations.RemoveAt(0);
				}
			
				return tmp;
			}
		}
		
		/// <summary>
		/// Helper function that converts a typed list into another type
		/// </summary>
		/// <returns>TA converted sequence</returns>
		/// <param name="input">The input list</param>
		/// <typeparam name="T">The type of the data in the returned enumerable.</typeparam>
		public static IEnumerable<T> ConvertList<T>(System.Collections.IEnumerable input)
		{
            return input.Cast<T>();
		}
		
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
        protected static readonly System.Reflection.MethodInfo binaryBaseMethodType = typeof(UFunc.ApplyManager).GetMethod("ApplyBinaryOp", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo unaryBaseMethodType = typeof(UFunc.ApplyManager).GetMethod("ApplyUnaryOp", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo nullaryBaseMethodType = typeof(UFunc.ApplyManager).GetMethod("ApplyNullaryOp", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo reduceBaseMethodType = typeof(UFunc.ApplyManager).GetMethod("ApplyReduce", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo matmulBaseMethodType = typeof(UFunc.ApplyManager).GetMethod("ApplyMatmul", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo aggregateBaseMethodType = typeof(UFunc.ApplyManager).GetMethod("ApplyAggregate", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
		/// Cache of the generic template method
		/// </summary>
		protected static readonly System.Reflection.MethodInfo unaryConversionBaseMethodType = typeof(UFunc.ApplyManager).GetMethod("ApplyUnaryConvOp", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of the generic template method
        /// </summary>
        protected static readonly System.Reflection.MethodInfo binaryConversionBaseMethodType = typeof(UFunc.ApplyManager).GetMethod("ApplyBinaryConvOp", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        /// <summary>
        /// Cache of instantiated template methods
        /// </summary>
        protected static readonly Dictionary<object, System.Reflection.MethodInfo> specializedMethods = new Dictionary<object, System.Reflection.MethodInfo>();
        /// <summary>
        /// Cache of instantiated template methods
        /// </summary>
        protected static readonly Dictionary<object, System.Reflection.MethodInfo> specializedReduceMethods = new Dictionary<object, System.Reflection.MethodInfo>();
        /// <summary>
        /// Cache of instantiated template methods
        /// </summary>
        protected static readonly Dictionary<object, System.Reflection.MethodInfo> specializedAggregateMethods = new Dictionary<object, System.Reflection.MethodInfo>();

		/// <summary>
		/// The clock required to get this accessor
		/// </summary>
       	protected long m_clock;

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
			m_clock = LazyAccessorCollector.AddOperation(new PendingOperation<T>(operation, operands));
        }

		/// <summary>
        /// Register a pending conversion operation on the underlying array
        /// </summary>
        /// <param name="operation">The operation performed</param>
        /// <param name="output">The output operand</param>
        /// <param name="input">The input operand</param>
        public virtual void AddConversionOperation<Ta>(IUnaryConvOp<Ta, T> operation, NdArray<T> output, NdArray<Ta> input)
		{
			m_clock = LazyAccessorCollector.AddOperation(new PendingUnaryConversionOperation<T, Ta>(operation, output, input));
        }

        /// <summary>
        /// Register a pending conversion operation on the underlying array
        /// </summary>
        /// <param name="operation">The operation performed</param>
        /// <param name="output">The output operand</param>
        /// <param name="in1">An input operand</param>
        /// <param name="in2">An input operand</param>
        public virtual void AddConversionOperation<Ta>(IBinaryConvOp<Ta, T> operation, NdArray<T> output, NdArray<Ta> in1, NdArray<Ta> in2)
		{
			m_clock = LazyAccessorCollector.AddOperation(new PendingBinaryConversionOperation<T, Ta>(operation, output, in1, in2));
        }

        /// <summary>
        /// Execute all operations that are pending to obtain the result array
        /// </summary>
        protected virtual void ExecutePendingOperations()
        {
            var lst = UnrollWorkList(this.m_clock);
            if (lst.Count > 0)
            	ExecuteOperations(lst);
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
        /// <param name="maxclock">The maximum clock to extract</param>
        /// <returns>A list of operations to perform</returns>
        public virtual IList<IPendingOperation> UnrollWorkList(long maxclock)
		{
			return LazyAccessorCollector.ExtractUntilClock(maxclock);
        }

        /// <summary>
        /// Basic execution function, simply calls the UFunc*Flush functions with the pending operation
        /// </summary>
        /// <param name="work">The list of operations to perform</param>
        public virtual void ExecuteOperations(IList<IPendingOperation> work)
        {
            DoExecute(work);
        }

		/// <summary>
		/// Basic execution function, simply calls the UFunc*Flush functions with the pending operation
		/// </summary>
		/// <param name="work">The list of operations to perform</param>
		public virtual void DoExecute(IList<IPendingOperation> work)
        {
            var tmp = new List<IPendingOperation>();
            while (work.Count > 0)
            {
                var pendingOpType = typeof(PendingOperation<>).MakeGenericType(new Type[] { work[0].DataType });
                var enumType = typeof(IEnumerable<>).MakeGenericType(new Type[] { pendingOpType });

                while (work.Count > 0 && (tmp.Count == 0 || work[0].TargetOperandType == tmp[0].TargetOperandType))
                {
                    tmp.Add(work[0]);
                    work.RemoveAt(0);
                }
				
                var cnvmethod = typeof(LazyAccessorCollector).GetMethod("ConvertList", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.DeclaredOnly | System.Reflection.BindingFlags.Static, null, new Type[] { typeof(System.Collections.IEnumerable) }, null).MakeGenericMethod(new Type[] { pendingOpType });
                var typedEnum = cnvmethod.Invoke(null, new object[] { tmp });
                var method = tmp[0].TargetOperandType.GetMethod("DoExecute", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.FlattenHierarchy | System.Reflection.BindingFlags.Instance, null, new Type[] { enumType }, null);
                method.Invoke(tmp[0].TargetAccessor, System.Reflection.BindingFlags.InvokeMethod, null, new object[] { typedEnum }, null);				
				tmp.Clear();
			}
		}
        
        /// <summary>
        /// Basic execution function, simply calls the UFunc*Flush functions with the pending operation
        /// </summary>
        /// <param name="work">The list of operations to perform</param>
        public virtual void DoExecute(IEnumerable<PendingOperation<T>> work)
        {
            foreach (var n in work)
            {
                if (n is IPendingBinaryConversionOp)
                {
                    Type inputType = n.GetType().GetGenericArguments()[1];

                    System.Reflection.MethodInfo genericVersion;
                    if (!specializedMethods.TryGetValue(n.Operation, out genericVersion))
                    {
                        genericVersion = binaryConversionBaseMethodType.MakeGenericMethod(inputType, typeof(T), n.Operation.GetType());
                        specializedMethods[n.Operation] = genericVersion;
                    }

                    genericVersion.Invoke(null, new object[] { n.Operation, ((IPendingUnaryConversionOp)n).InputOperand, ((IPendingBinaryConversionOp)n).InputOperand, n.Operands[0] });
                }
                else if (n is IPendingUnaryConversionOp)
                {
                    Type inputType = n.GetType().GetGenericArguments()[1];

                    System.Reflection.MethodInfo genericVersion;
                    if (!specializedMethods.TryGetValue(n.Operation, out genericVersion))
                    {
                        genericVersion = unaryConversionBaseMethodType.MakeGenericMethod(inputType, typeof(T), n.Operation.GetType());
                        specializedMethods[n.Operation] = genericVersion;
                    }

                    genericVersion.Invoke(null, new object[] { n.Operation, ((IPendingUnaryConversionOp)n).InputOperand, n.Operands[0] });
                }
                else if (n.Operation is NumCIL.UFunc.LazyReduceOperation<T>)
                {
                    NumCIL.UFunc.LazyReduceOperation<T> lzop = (NumCIL.UFunc.LazyReduceOperation<T>)n.Operation;

                    System.Reflection.MethodInfo genericVersion;
                    if (!specializedReduceMethods.TryGetValue(lzop.Operation.GetType(), out genericVersion))
                    {
                        genericVersion = reduceBaseMethodType.MakeGenericMethod(typeof(T), lzop.Operation.GetType());
                        specializedReduceMethods[lzop.Operation.GetType()] = genericVersion;
                    }

                    genericVersion.Invoke(null, new object[] { lzop.Operation, lzop.Axis, n.Operands[1], n.Operands[0] });

                }
                else if (n.Operation is NumCIL.UFunc.LazyMatmulOperation<T>)
                {
                    NumCIL.UFunc.LazyMatmulOperation<T> lzmt = (NumCIL.UFunc.LazyMatmulOperation<T>)n.Operation;

                    System.Reflection.MethodInfo genericVersion = matmulBaseMethodType.MakeGenericMethod(typeof(T), lzmt.AddOperator.GetType(), lzmt.MulOperator.GetType());
                    genericVersion.Invoke(null, new object[] { lzmt.AddOperator, lzmt.MulOperator, n.Operands[1], n.Operands[2], n.Operands[0] });

                }
                else if (n.Operation is NumCIL.UFunc.LazyAggregateOperation<T>)
                {
                    NumCIL.UFunc.LazyAggregateOperation<T> lzop = (NumCIL.UFunc.LazyAggregateOperation<T>)n.Operation;

                    System.Reflection.MethodInfo genericVersion;
                    if (!specializedAggregateMethods.TryGetValue(lzop.Operation.GetType(), out genericVersion))
                    {
                        genericVersion = aggregateBaseMethodType.MakeGenericMethod(typeof(T), lzop.Operation.GetType());
                        specializedAggregateMethods[lzop.Operation.GetType()] = genericVersion;
                    }
                    
                    // Store the parameters so we can access the return value by-ref
                    var paramlist = new object[] { lzop.Operation, n.Operands[1], default(T) };
                    genericVersion.Invoke(null, paramlist);
                    n.Operands[0].Value[0] = (T)paramlist[2];
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
    }

	/// <summary>
	/// Marker interface for a pending operation
	/// </summary>
	public interface IPendingOperation
	{
        /// <summary>
        /// Gets the clock.
        /// </summary>
		long Clock { get; }
        /// <summary>
        /// Gets the type of the target operand.
        /// </summary>
		Type TargetOperandType { get; }
        /// <summary>
        /// Gets the type of the data.
        /// </summary>
		Type DataType { get; }
        /// <summary>
        /// Gets the target accessor.
        /// </summary>
		object TargetAccessor { get; }
	}

    /// <summary>
    /// Representation of a pending operation
    /// </summary>
    /// <typeparam name="T">The type of data in the array</typeparam>
    public class PendingOperation<T> : IPendingOperation
    {
        /// <summary>
        /// The relative time this operation was registered
        /// </summary>
        private readonly long m_clock;
        
		/// <summary>
		/// The relative time this operation was registered
		/// </summary>
		public long Clock { get { return m_clock; } }
        
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
        /// Constructs a new pending operation
        /// </summary>
        /// <param name="operation">The operation to perform</param>
        /// <param name="operands">The operands involved</param>
        public PendingOperation(IOp<T> operation, params NdArray<T>[] operands)
        {
            this.m_clock = LazyAccessorCollector.ClockTick;
            this.Operation = operation;
            this.Operands = operands;
        }
        
        /// <summary>
        /// Gets the type of the target operand
        /// </summary>
        /// <value>The type of the target operand.</value>
        public Type TargetOperandType
        {
        	get
        	{
        		return Operands[0].DataAccessor.GetType();
        	}
        }
        
        /// <summary>
        /// Gets the type of the data
        /// </summary>
        /// <value>The type of the data.</value>
        public Type DataType
        {
        	get
        	{
        		return typeof(T);
        	}
        }

        /// <summary>
        /// Gets the target accessor.
        /// </summary>
        /// <value>The target accessor.</value>
        public object TargetAccessor
        {
        	get
        	{
        		return Operands[0].DataAccessor;
        	}
        }
    }

	/// <summary>
	/// Marker interface for quick recognition of conversion operations
	/// </summary>
	public interface IPendingUnaryConversionOp : IPendingOperation
	{
        /// <summary>
        /// Gets the untyped input operand
        /// </summary>
		object InputOperand { get; }
	}

    /// <summary>
    /// Marker interface for quick recognition of conversion operations
    /// </summary>
	public interface IPendingBinaryConversionOp : IPendingOperation
    {
        /// <summary>
        /// Gets the untyped input operand
        /// </summary>
        object InputOperand { get; }
    }

	/// <summary>
	/// Representation of a pending unary conversion operation.
	/// </summary>
	public class PendingUnaryConversionOperation<Ta, Tb> : PendingOperation<Ta>, IPendingUnaryConversionOp
	{
		/// <summary>
		/// The first input operand.
		/// </summary>
		public readonly NdArray<Tb> InputOperand;

        /// <summary>
        /// Constructs a new pending unary operation
        /// </summary>
        /// <param name="operation">The operation to perform</param>
        /// <param name="output">The output operand</param>
        /// <param name="input">The input operand</param>
        public PendingUnaryConversionOperation(IOp<Ta> operation, NdArray<Ta> output, NdArray<Tb> input)
            : base(operation, output)
        {
            InputOperand = input;
        }


		#region IPendingUnaryConversionOp implementation
		/// <summary>
		/// Gets the input operand as an untyped object.
		/// </summary>
		object IPendingUnaryConversionOp.InputOperand
		{
			get
			{
				return InputOperand;
			}
		}
		#endregion

	}

    /// <summary>
	/// Representation of a pending binary conversion operation.
	/// </summary>
    public class PendingBinaryConversionOperation<Ta, Tb> : PendingUnaryConversionOperation<Ta, Tb>, IPendingBinaryConversionOp
    {
        /// <summary>
        /// The first input operand.
        /// </summary>
        public readonly NdArray<Tb> InputOperandRhs;

        /// <summary>
        /// Constructs a new pending binary operation
        /// </summary>
        /// <param name="operation">The operation to perform</param>
        /// <param name="output">The output operand</param>
        /// <param name="in1">An input operand</param>
        /// <param name="in2">An input operand</param>
        public PendingBinaryConversionOperation(IOp<Ta> operation, NdArray<Ta> output, NdArray<Tb> in1, NdArray<Tb> in2)
            :base(operation, output, in1)
        {
            InputOperandRhs = in2;
        }

        #region IPendingBinaryConversionOp implementation
        /// <summary>
        /// Gets the input operand as an untyped object.
        /// </summary>
        object IPendingBinaryConversionOp.InputOperand
        {
            get { return this.InputOperandRhs; }
        }
        #endregion
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
