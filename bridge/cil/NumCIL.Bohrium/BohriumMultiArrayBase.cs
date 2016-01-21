using System;
using System.Linq;

namespace NumCIL.Bohrium
{
	internal interface IMultiArray : IDisposable
	{
		/// <summary>
		/// Gets the underlying multi array pointer
		/// </summary>
		IntPtr Pointer { set; get; }

		/// <summary>
		/// Issues a sync on this instance
		/// </summary>
		void Sync();
	}
		
	internal interface ITypedMultiArrayMapper<TPointer>
	{
		bool IsAllocated(TPointer self);
		TPointer NewView(TPointer self, ulong rank, long offset, long[] dimensions, long[] stride);
		TPointer NewEmpty(ulong size);
		void Destroy(TPointer self);
		void Sync(TPointer self);
		void Copy(TPointer self, TPointer source);
		IntPtr GetData(TPointer self);
		void SetData(TPointer self, IntPtr data);
	}

	/// <summary>
	/// The multi-array instance
	/// </summary>
	internal static class BohriumMultiArrayBase<TData, TPointer, TAccess>
		where TAccess : ITypedMultiArrayMapper<TPointer>, new()
		where TPointer : IMultiArray
	{
		/// <summary>
		/// The static mapper accessor instance
		/// </summary>
		public static readonly ITypedMultiArrayMapper<TPointer> Mapper = new TAccess();

		/// <summary>
		/// Checks if the array is allocated, i.e. not null
		/// </summary>
		public static bool IsAllocated(TPointer self)
		{
			return Mapper.IsAllocated(self);
		}

		/// <summary>
		/// Creates a new view
		/// </summary>
		public static TPointer CreateNewView(NumCIL.Generic.NdArray<TData> a)
		{
			var ac = a.DataAccessor;
			lock(PinnedArrayTracker.ExecuteLock)
			{
				if (ac is BohriumDataAccessorBase<TData, TPointer, TAccess>) {
					var acp = ((BohriumDataAccessorBase<TData, TPointer, TAccess>)a.DataAccessor).GetArrayPointer();


					return Mapper.NewView(
						acp,
						(ulong)a.Shape.Dimensions.Length, 
						a.Shape.Offset, 
						a.Shape.Dimensions.Select(x => x.Length).ToArray(),
						a.Shape.Dimensions.Select(x => x.Stride).ToArray()
					);
				} else {
					TPointer res;
					using(var acp = Mapper.NewEmpty((ulong)a.DataAccessor.Length))
					{
						res = Mapper.NewView(
							acp,
							(ulong)a.Shape.Dimensions.Length, 
							a.Shape.Offset, 
							a.Shape.Dimensions.Select(x => x.Length).ToArray(),
							a.Shape.Dimensions.Select(x => x.Stride).ToArray()
						);
					}

					Mapper.SetData(res, PinnedArrayTracker.CreatePinnedArray(a.DataAccessor.AsArray()));
					return res;
				}

				//Console.WriteLine("Created multi_array from NdArray: {0}", this.pointer.ToInt64());
			}
		}

		/// <summary>
		/// Releases the memory reserved by the pointer
		/// </summary>
		public static void Dispose(TPointer self)
		{
			try 
			{ 
				if (Mapper.IsAllocated(self))
				{
					//Console.WriteLine("Destroying multi_array {0}", this.pointer.ToInt64());

					var data = GetData(self);
					lock(PinnedArrayTracker.ExecuteLock)
					{
						if (PinnedArrayTracker.DecReference(data))
							Mapper.SetData(self, IntPtr.Zero);
						
						Mapper.Destroy(self);
					}
				}
			}
			finally 
			{ 
				self.Pointer = IntPtr.Zero; 
			}
		}

		/// <summary>
		/// Issues a synchronization for the underlying data
		/// </summary>
		public static void Sync(TPointer self)
		{
			lock (PinnedArrayTracker.ExecuteLock)
				Mapper.Sync(self);
		}

		/// <summary>
		/// Returns the multi_array as a string
		/// </summary>
		public static string AsString(TPointer self)
		{
			var s = self.Pointer.ToInt64();
			return string.Format("self: {0} (d: {1})", s, GetData(self).ToInt64());
		}
			
		/// <summary>
		/// Gets the data pointer
		/// </summary>
		public static IntPtr GetData(TPointer self)
		{ 
			lock (PinnedArrayTracker.ExecuteLock)
				return Mapper.GetData(self);
		}

		/// <summary>
		/// Sets the data pointer
		/// </summary>
		/// <param name="self">Self.</param>
		/// <param name="value">Value.</param>
		public static void SetData(TPointer self, IntPtr value)
		{ 
			var prevPtr = GetData(self);
			if (prevPtr != value)
			{
				PinnedArrayTracker.DecReference(prevPtr);
				PinnedArrayTracker.IncReference(value);
				lock(PinnedArrayTracker.ExecuteLock)
				{
					Mapper.SetData(self, value); 
					/*if (PinnedArrayTracker.IsManagedData(value))
						Mapper.SetExternal(self, true);*/
				}
			}
		}
	}
}

