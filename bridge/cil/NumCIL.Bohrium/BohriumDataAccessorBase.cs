using System;

namespace NumCIL.Bohrium
{
	public interface IBohriumAccessor
	{
		void SetDirty();
	}

	internal abstract class BohriumDataAccessorBase<TData, TPointer, TAccess> : NumCIL.Generic.IDataAccessor<TData>, IDisposable, NumCIL.Generic.IFlushableAccessor, IBohriumAccessor
		where TAccess : ITypedMultiArrayMapper<TPointer>, new()
		where TPointer : IDisposable
	{
		/// <summary>
		/// The static mapper accessor instance
		/// </summary>
		public static readonly ITypedMultiArrayMapper<TPointer> Mapper = new TAccess();

		protected TPointer m_array;
		protected TData[] m_data;
		protected readonly long m_size;
		protected bool m_dirty;
		private static readonly int ELSIZE = System.Runtime.InteropServices.Marshal.SizeOf(typeof(TData));
		private static long MAX_INDEX = int.MaxValue / ELSIZE;

		protected abstract TPointer NewFromValue(TData value, ulong size);
		protected abstract TPointer NewFromEmpty(ulong size);

		public BohriumDataAccessorBase(TData value, long size)
		{
			m_size = size;
			m_array = NewFromValue(value, (ulong)size);
		}

		public BohriumDataAccessorBase(long size)
		{
			m_size = size;
			//Special case, we do not allocate until the first usage
		}

		public BohriumDataAccessorBase(TData[] data)
		{
			if (data == null)
				throw new ArgumentNullException("data");

			m_data = data;
			m_size = data.Length;

			//The array and handle are not allocated unless needed
		}

		public TPointer MultiArray
		{
			get { return m_array; }
			set
			{
				if (m_data == null && ! Mapper.IsAllocated(m_array))
					m_array = value;
				else
					throw new InvalidOperationException("Cannot set array on allocated instance");
			}
		}

		#region IDataAccessor implementation

		public TData[] AsArray()
		{
			Allocate();

			return m_data;
		}

		public void SetDirty()
		{
			m_dirty = true;
		}

		internal TPointer GetArrayPointer()
		{
			if (!Mapper.IsAllocated(m_array))
			{
				lock(PinnedArrayTracker.ExecuteLock)
				{
					m_array = NewFromEmpty((ulong)m_size);
					if (m_data == null)
					{
						//Console.WriteLine("Creating data in C land");

						// No CIL allocation, create a new multi-array
						m_dirty = true;
					}
					else
					{
						//Console.WriteLine("Wraping CIL data");

						// Existing CIL allocation, wrap it
						Mapper.SetData(m_array, PinnedArrayTracker.CreatePinnedArray(m_data));
					}
				}
			}

			return m_array;
		}

		public void Flush()
		{
			Sync();
		}

		private void Sync()
		{
			if (m_dirty)
			{
				// Console.WriteLine("Calling sync");
				if (!Mapper.IsAllocated(m_array))
					throw new InvalidOperationException("Array cannot be dirty and not allocated");

				Mapper.Sync(m_array);
				PinnedArrayTracker.Release();
				m_dirty = false;
			}
		}

		public void Allocate()
		{
			if (Mapper.IsAllocated(m_array))
			{
				if (m_data == null)
				{
					//Console.WriteLine("Copy data from C land");

					//In this case, data resides in C land, 
					// so we copy it back to CIL and free it in C land
					m_data = new TData[m_size];

					using (var t = NewFromEmpty((ulong)m_size))
					{
						Mapper.SetData(t, PinnedArrayTracker.CreatePinnedArray(m_data));
						Mapper.Copy(t, m_array);
						Mapper.Sync(t);
						m_array.Dispose();
						PinnedArrayTracker.Release();
					}

					m_dirty = false;
				}
				else
				{
					//Data resides in CIL so detach the data and release the array
					//Console.WriteLine("Release data in C land");
					Sync();
					Mapper.SetData(m_array, IntPtr.Zero);
					m_array.Dispose();
				}
			}
			else
			{
				//Console.WriteLine("Accessed array that was never in any operation");
				Sync();
				if (m_data == null)
					m_data = new TData[m_size];
			}
		}

		public long Length
		{
			get
			{
				return m_size;
			}
		}

		public bool IsAllocated
		{
			get
			{
				return m_data != null;
			}
		}

		public TData this[long index]
		{
			get
			{
				Sync();

				if (m_data == null && Mapper.IsAllocated(m_array) && index < MAX_INDEX)
				{
					if (index < 0 || index >= m_size)
						throw new ArgumentOutOfRangeException("index");

					//Console.WriteLine("In get, using pointer");

					//Console.WriteLine("array is {0}", m_array.AsString());

					var ptr = Mapper.GetData(m_array);
					if (ptr == IntPtr.Zero)
						throw new InvalidOperationException("The data pointer was null");

					var tmp = new TData[1];
					Utility.WritePointerToArray(ptr + (int)(index * ELSIZE), tmp);
					return tmp[0];
				}

				if (m_data == null)
					Allocate();

				//Console.WriteLine("In get, using local data");
				return m_data[index];
			}
			set
			{
				Sync();

				if (m_data == null && Mapper.IsAllocated(m_array) && index < MAX_INDEX)
				{
					if (index < 0 || index >= m_size)
						throw new ArgumentOutOfRangeException("index");

					var ptr = Mapper.GetData(m_array);
					if (ptr == IntPtr.Zero)
						throw new InvalidOperationException("The data pointer was null");

					var tmp = new TData[] { value };
					Utility.WriteArrayToPointer(tmp, ptr + (int)(index * ELSIZE));
					return;
				}

				if (m_data == null)
					Allocate();

				m_data[index] = value;
			}
		}

		public object Tag { get; set; }


		#endregion

		/// <summary>
		/// Releases the memory reserved by the pointer
		/// </summary>
		public void Dispose()
		{
			Dispose(true);
		}

		/// <summary>
		/// Releases the memory reserved by the pointer
		/// </summary>
		~BohriumDataAccessorBase()
		{
			Dispose(false);
		}

		/// <summary>
		/// Releases the memory reserved by the pointer
		/// </summary>
		private void Dispose(bool disposing)
		{
			if (disposing)
				GC.SuppressFinalize(this);

			if (Mapper.IsAllocated(m_array))
				m_array.Dispose();
		}
	}    }

