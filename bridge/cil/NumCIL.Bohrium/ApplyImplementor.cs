using System;
using System.Linq;
using System.Collections.Generic;
using NumCIL.Generic;

using System.Reflection;
using NumCIL.Generic.Operators;

namespace NumCIL.Bohrium
{    
    public class ApplyImplementor : UFunc.IApplyHandler
    {
		public static Action<Type, Type[]> DEBUG_FALLBACK = (a, b) => { DEBUG_FALLBACK_STR(a.FullName, b); };

		public static Action<string, Type[]> DEBUG_FALLBACK_STR = (a, b) => 
#if DEBUG
			Console.WriteLine("*** Unhandled op {0} for types [ {1} ]", a, string.Join(",", b.Select(n => n.ToString())));
#else
			{ };
#endif

        
        #region IApplyHandler implementation
        public bool ApplyBinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out) 
            where C : struct, IBinaryOp<T>
        {
			return ApplyBinaryConvOp<T, T, C>(op, in1, in2, @out);
        }

        public bool ApplyUnaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> @out) 
			where C : struct, IUnaryOp<T>
        {
			return ApplyUnaryConvOp<T, T, C>(op, in1, @out);
        }

		public bool ApplyBinaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out) 
			where C : struct, IBinaryConvOp<Ta, Tb>
		{
			var opname = NameMapper.GetOperationName<C, Ta>();
			return ApplyBinaryConvOp<Ta, Tb>(opname, in1, in2, @out);
		}

        public bool ApplyBinaryConvOp<Ta, Tb>(string opname, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out) 
        {
			var isUnmanaged = @out.DataAccessor is IBohriumAccessor;                  

			var typenamea = NameMapper.GetTypeName<Ta>();
			var typenameb = NameMapper.GetTypeName<Tb>();
			var ptypea = NameMapper.GetPointerType<Ta>();
			var ptypeb = NameMapper.GetPointerType<Tb>();

			string methodname;
			MethodInfo method;

			// If in1 is a scalar value and there is a scalar accepting method, use that
			if (@in1.DataAccessor.GetType() == typeof(DefaultAccessor<Ta>) && @in1.DataAccessor.Length == 1)
			{
				methodname = string.Format("bhc_{0}_A{2}_K{1}_A{1}", opname, typenamea, typenameb);
				method = typeof(PInvoke).GetMethod(methodname, new Type[] { ptypeb, typeof(Ta), ptypea });
				if (method != null)
				{
					lock(PinnedArrayTracker.ExecuteLock)
					{
						using (var v2 = @in2.WrapWithPointer())
						using (var v0 = @out.WrapWithPointer())
						{
							method.Invoke(null, new object[] { v0, in1.Value[0], v2 });

							if (isUnmanaged)
								((IBohriumAccessor)@out.DataAccessor).SetDirty();
							else
								v0.Sync();
						}
					}

					// If the output is CIL-managed, we must sync immediately
					if (!isUnmanaged)
						PinnedArrayTracker.Release();

					return true;
				}
			}

			// If in2 is a scalar value and there is a scalar accepting method, use that
			if (@in2.DataAccessor.GetType() == typeof(DefaultAccessor<Ta>) && @in2.DataAccessor.Length == 1)
			{
				methodname = string.Format("bhc_{0}_A{2}_A{1}_K{1}", opname, typenamea, typenameb);
				method = typeof(PInvoke).GetMethod(methodname, new Type[] { ptypeb, ptypea, typeof(Ta) });
				if (method != null)
				{
					lock(PinnedArrayTracker.ExecuteLock)
					{
						using (var v1 = @in1.WrapWithPointer())
						using (var v0 = @out.WrapWithPointer())
						{
							method.Invoke(null, new object[] { v0, v1, in2.Value[0] });

							if (isUnmanaged)
								((IBohriumAccessor)@out.DataAccessor).SetDirty();
							else
								v0.Sync();
						}
					}

					// If the output is CIL-managed, we must sync immediately
					if (!isUnmanaged)
						PinnedArrayTracker.Release();

					return true;
				}
			}

			// Default, the all-array method
			methodname = string.Format("bhc_{0}_A{2}_A{1}_A{1}", opname, typenamea, typenameb);
			method = typeof(PInvoke).GetMethod(methodname, new Type[] { ptypeb, ptypea, ptypea });
			if (method != null)
			{
				lock(PinnedArrayTracker.ExecuteLock)
				{
					using (var v1 = @in1.WrapWithPointer())
					using (var v2 = @in2.WrapWithPointer())
					using (var v0 = @out.WrapWithPointer())
					{
						method.Invoke(null, new object[] { v0, v1, v2 });

						if (isUnmanaged)
							((IBohriumAccessor)@out.DataAccessor).SetDirty();
						else
							v0.Sync();
					}
				}

				// If the output is CIL-managed, we must sync immediately
				if (!isUnmanaged)
					PinnedArrayTracker.Release();

				return true;
			}

			DEBUG_FALLBACK_STR(opname, new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
			return false;        
		}

        public bool ApplyUnaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out) where C : struct, IUnaryConvOp<Ta, Tb>
        {
			var isUnmanaged = @out.DataAccessor is IBohriumAccessor;                  

			var typenamea = NameMapper.GetTypeName<Ta>();
			var typenameb = NameMapper.GetTypeName<Tb>();
			var opname = NameMapper.GetOperationName<C, Ta>();
			var ptypea = NameMapper.GetPointerType<Ta>();
			var ptypeb = NameMapper.GetPointerType<Tb>();

			string methodname;
			MethodInfo method;

			// If in1 is a scalar value and there is a scalar accepting method, use that
			if (@in1.DataAccessor.GetType() == typeof(DefaultAccessor<Ta>) && @in1.DataAccessor.Length == 1)
			{
				methodname = string.Format("bhc_{0}_A{2}_K{1}", opname, typenamea, typenameb);
				method = typeof(PInvoke).GetMethod(methodname, new Type[] { ptypeb, typeof(Ta) });
				if (method != null)
				{
					lock(PinnedArrayTracker.ExecuteLock)
					{
						using (var v0 = @out.WrapWithPointer())
						{
							method.Invoke(null, new object[] { v0, in1.Value[0] });

							if (isUnmanaged)
								((IBohriumAccessor)@out.DataAccessor).SetDirty();
							else
								v0.Sync();
						}
					}

					// If the output is CIL-managed, we must sync immediately
					if (!isUnmanaged)
						PinnedArrayTracker.Release();

					return true;
				}
			}				

			// Default, the all-array method
			methodname = string.Format("bhc_{0}_A{2}_A{1}", opname, typenamea, typenameb);
			method = typeof(PInvoke).GetMethod(methodname, new Type[] { ptypeb, ptypea });
			if (method != null)
			{
				lock(PinnedArrayTracker.ExecuteLock)
				{
					using (var v1 = @in1.WrapWithPointer())
					using (var v0 = @out.WrapWithPointer())
					{
						method.Invoke(null, new object[] { v0, v1 });

						if (isUnmanaged)
							((IBohriumAccessor)@out.DataAccessor).SetDirty();
						else
							v0.Sync();
					}
				}

				// If the output is CIL-managed, we must sync immediately
				if (!isUnmanaged)
					PinnedArrayTracker.Release();

				return true;
			}

			DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
			return false;        
		}

        public bool ApplyNullaryOp<T, C>(C op, NdArray<T> @out) where C : struct, INullaryOp<T>
        {
			var isUnmanaged = @out.DataAccessor is IBohriumAccessor;

			var typename = NameMapper.GetTypeName<T>();
			var opname = NameMapper.GetOperationName<C, T>();
			var ptype = NameMapper.GetPointerType<T>();

			var methodname = string.Format("bhc_{0}_A{1}", opname, typename);
			var method = typeof(PInvoke).GetMethod(methodname, new Type[] { ptype });
			if (method != null)
			{
				lock(PinnedArrayTracker.ExecuteLock)
				{
					using (var v0 = @out.WrapWithPointer())
					{
						method.Invoke(null, new object[] { v0 });

						if (isUnmanaged)
							((IBohriumAccessor)@out.DataAccessor).SetDirty();
						else
							v0.Sync();
					}
				}

				// If the output is CIL-managed, we must sync immediately
				if (!isUnmanaged)
					PinnedArrayTracker.Release();

				return true;
			}

			// Random and range are supplied by mapping the ulong version
			if (op is IRandomGeneratorOp<T>)
			{
				if (RandomSuppliers.Random<T>(@out))
					return true;
			}
			else if (op is IRangeGeneratorOp<T>)
			{
				if (RangeSuppliers.Range<T>(@out))
					return true;
			}


			DEBUG_FALLBACK(op.GetType(), new Type[] {@out.DataAccessor.GetType(), @out.DataAccessor.GetType() });
			return false;        
		}

        public bool ApplyReduce<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out) where C : struct, IBinaryOp<T>
        {
			var isUnmanaged = @out.DataAccessor is IBohriumAccessor;                  

			var typename = NameMapper.GetTypeName<T>();
			var opname = NameMapper.GetOperationName<C, T>();
			var ptype = NameMapper.GetPointerType<T>();

			// Default, the all-array method
			var methodname = string.Format("bhc_{0}_reduce_A{2}_A{1}_Kint64", opname, typename, typename);
			var method = typeof(PInvoke).GetMethod(methodname, new Type[] { ptype, ptype, typeof(long) });
			if (method != null)
			{
				lock(PinnedArrayTracker.ExecuteLock)
				{
					using (var v1 = @in1.WrapWithPointer())
					using (var v0 = @out.WrapWithPointer())
					{
						method.Invoke(null, new object[] { v0, v1, axis });

						if (isUnmanaged)
							((IBohriumAccessor)@out.DataAccessor).SetDirty();
						else
							v0.Sync();
					}
				}

				// If the output is CIL-managed, we must sync immediately
				if (!isUnmanaged)
					PinnedArrayTracker.Release();

				return true;
			}
				
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyMatmul<T, CADD, CMUL>(CADD addop, CMUL mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null) where CADD : struct, IBinaryOp<T> where CMUL : struct, IBinaryOp<T>
        {
			if (typeof(IAdd).IsAssignableFrom(typeof(CADD)) && typeof(IMul).IsAssignableFrom(typeof(CMUL)))
				return ApplyBinaryConvOp<T, T>("matmul", in1, in2, @out);

			DEBUG_FALLBACK_STR("matmul", new Type[] {typeof(CADD), typeof(CMUL), in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
			return false;
        }

        public bool ApplyAggregate<T, C>(C op, NdArray<T> in1, out T result) where C : struct, IBinaryOp<T>
        {
			// For non-1D, map into a 1D array
			if (in1.Shape.Dimensions.Length != 1)
			{
				// If the source is contiguos, run reduce with flattened 1D array
				if (in1.Shape.IsPlain)
				{
					in1 = in1.Reshape(new Shape(new long[] { in1.Shape.Elements }, in1.Shape.Offset, new long[] { 1 }));
				}
				else
				{
					// Allocate a temporary placeholder and reduce the original into the temporary
					var tmp = new NdArray<T>(new Shape(in1.Shape.Dimensions.Take(in1.Shape.Dimensions.Length - 1).Select(x => x.Length).ToArray()));
					if (!ApplyReduce<T, C>(op, 0, in1, tmp))
					{
						DEBUG_FALLBACK_STR("reduce", new Type[] { op.GetType(), in1.DataAccessor.GetType(), typeof(T) });
						result = default(T);
						return false;
					}

					// Treat the compact result as a 1D array
					in1 = tmp.Reshape(new long[] { tmp.Shape.Length });
				}
			}


			// Now that we have a 1D reduction, simply map to reduce call
			var target = new NdArray<T>(new Shape(1));
			if (!ApplyReduce<T, C>(op, 0, in1, target))
			{
				DEBUG_FALLBACK_STR("reduce", new Type[] { op.GetType(), in1.DataAccessor.GetType(), typeof(T) });
				result = default(T);
				return false;
			}

			result = target.DataAccessor[0];
			return true;
        }
        #endregion
        

    }    
}

