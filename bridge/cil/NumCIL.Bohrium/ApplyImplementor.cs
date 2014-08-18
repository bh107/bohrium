using System;
using System.Linq;
using System.Collections.Generic;
using NumCIL.Generic;

using bh_bool = System.Boolean;
using bh_int8 = System.SByte;
using bh_uint8 = System.Byte;
using bh_int16 = System.Int16;
using bh_uint16 = System.UInt16;
using bh_int32 = System.Int32;
using bh_uint32 = System.UInt32;
using bh_int64 = System.Int64;
using bh_uint64 = System.UInt64;
using bh_float32 = System.Single;
using bh_float64 = System.Double;
using bh_complex64 = NumCIL.Complex64.DataType;
using bh_complex128 = System.Numerics.Complex;

namespace NumCIL.Bohrium
{    
    public class ApplyImplementor : UFunc.IApplyHandler
    {
        public static Action<Type, Type[]> DEBUG_FALLBACK = (a, b) => Console.WriteLine("*** Unhandled op {0} for types [ {1} ]", a.FullName,string.Join(",", b.Select(n => n.ToString())));
        
        private static Dictionary<Type, object> _implementors = new Dictionary<Type, object>();
        
        static ApplyImplementor()
        {
            _implementors[typeof(bh_bool)] = new ApplyImplementor_bool8();
            _implementors[typeof(bh_int8)] = new ApplyImplementor_int8();
            _implementors[typeof(bh_uint8)] = new ApplyImplementor_uint8();
            _implementors[typeof(bh_int16)] = new ApplyImplementor_int16();
            _implementors[typeof(bh_uint16)] = new ApplyImplementor_uint16();
            _implementors[typeof(bh_int32)] = new ApplyImplementor_int32();
            _implementors[typeof(bh_uint32)] = new ApplyImplementor_uint32();
            _implementors[typeof(bh_int64)] = new ApplyImplementor_int64();
            _implementors[typeof(bh_uint64)] = new ApplyImplementor_uint64();
            _implementors[typeof(bh_float32)] = new ApplyImplementor_float32();
            _implementors[typeof(bh_float64)] = new ApplyImplementor_float64();
            _implementors[typeof(bh_complex64)] = new ApplyImplementor_complex64();
            _implementors[typeof(bh_complex128)] = new ApplyImplementor_complex128();
        }
        
        #region IApplyHandler implementation
        public bool ApplyBinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out) 
            where C : struct, IBinaryOp<T>
        {
            object h;
            if (_implementors.TryGetValue(typeof(T), out h))
            {
                var i = (ITypedApplyImplementor<T>)h;
                if (i.ApplyBinaryOp(op.GetType(), in1, in2, @out))
                    return true;
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
                if (i.ApplyUnaryOp(op.GetType(), in1, @out))
                    return true;
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
                if(i.ApplyBinaryConvOp<Ta>(op.GetType(), in1, in2, @out))
                    return true;
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
                if(i.ApplyUnaryConvOp<Ta>(op.GetType(), in1, @out))
                    return true;
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
                if (i.ApplyReduce(op.GetType(), axis, in1, @out))
                    return true;
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
                if (i.ApplyMatmul(addop.GetType(), mulop.GetType(), in1, in2, @out))
                    return true;
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
                if (i.ApplyAggregate(op.GetType(), in1, out result))
                    return true;
                
            }
            
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), typeof(T) });
            result = default(T);
            return false;
        }
        #endregion
        

    }    
}

