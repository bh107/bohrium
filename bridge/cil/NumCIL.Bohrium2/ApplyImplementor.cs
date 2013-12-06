using System;
using System.Linq;
using System.Collections.Generic;
using NumCIL.Generic;

namespace NumCIL.Bohrium2
{    
    public class ApplyImplementor : UFunc.IApplyHandler
    {
        public static Action<Type, Type[]> DEBUG_FALLBACK = (a, b) => Console.WriteLine("*** Unhandled op {0} for types [ {1} ]", a.FullName,string.Join(",", b.Select(n => n.ToString())));
        
        private static Dictionary<Type, object> _implementors = new Dictionary<Type, object>();
        
        static ApplyImplementor()
        {
            _implementors[typeof(float)] = new ApplyImplementor_float32();
        }
        
        #region IApplyHandler implementation
        public bool ApplyBinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out) 
            where C : struct, IBinaryOp<T>
        {
            object h;
            if (_implementors.TryGetValue(typeof(T), out h))
            {
                var i = (ITypedApplyImplementor<T>)h;
                return i.ApplyBinaryOp(op.GetType(), in1, in2, @out);
            }
            
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), in2.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyUnaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> @out) where C : struct, IUnaryOp<T>
        {
            object h;
            if (_implementors.TryGetValue(typeof(T), out h))
            {
                var i = (ITypedApplyImplementor<T>)h;
                return i.ApplyUnaryOp(op.GetType(), in1, @out);
            }
            
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyBinaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out) where C : struct, IBinaryConvOp<Ta, Tb>
        {
            object h;
            if (_implementors.TryGetValue(typeof(Tb), out h))
            {
                var i = (ITypedApplyImplementor<Tb>)h;
                return i.ApplyBinaryConvOp<Ta>(op.GetType(), in1, in2, @out);
            }
            
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), in2.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyUnaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out) where C : struct, IUnaryConvOp<Ta, Tb>
        {
            object h;
            if (_implementors.TryGetValue(typeof(Tb), out h))
            {
                var i = (ITypedApplyImplementor<Tb>)h;
                return i.ApplyUnaryConvOp<Ta>(op.GetType(), in1, @out);
            }
            
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyNullaryOp<T, C>(C op, NdArray<T> @out) where C : struct, INullaryOp<T>
        {
            object h;
            if (_implementors.TryGetValue(typeof(T), out h))
            {
                var i = (ITypedApplyImplementor<T>)h;
                return i.ApplyNullaryOp(op.GetType(), @out);
            }
            DEBUG_FALLBACK(op.GetType(), new Type[] { @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyReduce<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out) where C : struct, IBinaryOp<T>
        {
            object h;
            if (_implementors.TryGetValue(typeof(T), out h))
            {
                var i = (ITypedApplyImplementor<T>)h;
                return i.ApplyReduce(op.GetType(), axis, in1, @out);
            }
            
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyMatmul<T, CADD, CMUL>(CADD addop, CMUL mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null) where CADD : struct, IBinaryOp<T> where CMUL : struct, IBinaryOp<T>
        {
            object h;
            if (_implementors.TryGetValue(typeof(T), out h))
            {
                var i = (ITypedApplyImplementor<T>)h;
                return i.ApplyMatmul(addop.GetType(), mulop.GetType(), in1, in2, @out);
            }
            DEBUG_FALLBACK(addop.GetType(), new Type[] { mulop.GetType(), in1.DataAccessor.GetType(), in2.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyAggregate<T, C>(C op, NdArray<T> in1, out T result) where C : struct, IBinaryOp<T>
        {
            object h;
            if (_implementors.TryGetValue(typeof(T), out h))
            {
                var i = (ITypedApplyImplementor<T>)h;
                return i.ApplyAggregate(op.GetType(), in1, out result);
            }
            
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), typeof(T) });
            result = default(T);
            return false;
        }
        #endregion
        

    }    
}

