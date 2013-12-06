using System;
using System.Linq;
using System.Collections.Generic;
using NumCIL.Generic;

using multi32 = NumCIL.Bohrium2.PInvoke.bh_multi_array_float32_p;

namespace NumCIL.Bohrium2
{    
    public class ApplyImplementor : UFunc.IApplyHandler
    {
        public static Action<Type, Type[]> DEBUG_FALLBACK = (a, b) => Console.WriteLine("*** Unhandled op {0} for types [ {1} ]", a.FullName,string.Join(",", b.Select(n => n.ToString())));
    
        private Tuple<Type, Func<multi32, multi32>>[] m_unOps = 
        {
            new Tuple<Type, Func<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IAbs), PInvoke.bh_multi_array_float32_absolute),
            new Tuple<Type, Func<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IFloor), PInvoke.bh_multi_array_float32_floor),
            new Tuple<Type, Func<multi32, multi32>>(typeof(NumCIL.Generic.Operators.ICeiling), PInvoke.bh_multi_array_float32_ceil),
            new Tuple<Type, Func<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IRound), PInvoke.bh_multi_array_float32_rint),
            new Tuple<Type, Func<multi32, multi32>>(typeof(NumCIL.Generic.Operators.ISqrt), PInvoke.bh_multi_array_float32_sqrt),
        };
        
        private Tuple<Type, Func<multi32, multi32, multi32>>[] m_binOps = 
        {
            new Tuple<Type, Func<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IAdd), PInvoke.bh_multi_array_float32_add),
            new Tuple<Type, Func<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.ISub), PInvoke.bh_multi_array_float32_subtract),
            new Tuple<Type, Func<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMul), PInvoke.bh_multi_array_float32_multiply),
            new Tuple<Type, Func<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IDiv), PInvoke.bh_multi_array_float32_divide),
            new Tuple<Type, Func<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMod), PInvoke.bh_multi_array_float32_modulo),
            new Tuple<Type, Func<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMax), PInvoke.bh_multi_array_float32_maximum),
            new Tuple<Type, Func<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMin), PInvoke.bh_multi_array_float32_minimin),
        };
        
        private Dictionary<Type, Func<multi32, multi32, multi32>> m_binOpLookup = new Dictionary<Type, Func<multi32, multi32, multi32>>();
        private Dictionary<Type, Func<multi32, multi32>> m_unOpLookup = new Dictionary<Type, Func<multi32, multi32>>();
    
        #region IApplyHandler implementation
        public bool ApplyBinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out) 
            where C : struct, IBinaryOp<T>
        {
            if (typeof(T) == typeof(float))
                return ApplyBinaryOp_float32(op.GetType(), (NdArray<float>)(object)in1, (NdArray<float>)(object)in2, (NdArray<float>)(object)@out);

            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), in2.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyUnaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> @out) where C : struct, IUnaryOp<T>
        {
            if (typeof(T) == typeof(float))
                return ApplyUnaryOp_float32(op.GetType(), (NdArray<float>)(object)in1, (NdArray<float>)(object)@out);
            
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyBinaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<Tb> @out) where C : struct, IBinaryConvOp<Ta, Tb>
        {
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), in2.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyUnaryConvOp<Ta, Tb, C>(C op, NdArray<Ta> in1, NdArray<Tb> @out) where C : struct, IUnaryConvOp<Ta, Tb>
        {
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyNullaryOp<T, C>(C op, NdArray<T> @out) where C : struct, INullaryOp<T>
        {
            DEBUG_FALLBACK(op.GetType(), new Type[] { @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyReduce<T, C>(C op, long axis, NdArray<T> in1, NdArray<T> @out) where C : struct, IBinaryOp<T>
        {
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyMatmul<T, CADD, CMUL>(CADD addop, CMUL mulop, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out = null) where CADD : struct, IBinaryOp<T> where CMUL : struct, IBinaryOp<T>
        {
            DEBUG_FALLBACK(addop.GetType(), new Type[] { mulop.GetType(), in1.DataAccessor.GetType(), in2.DataAccessor.GetType(), @out.DataAccessor.GetType() });
            return false;
        }

        public bool ApplyAggregate<T, C>(C op, NdArray<T> in1, out T result) where C : struct, IBinaryOp<T>
        {
            DEBUG_FALLBACK(op.GetType(), new Type[] {in1.DataAccessor.GetType(), typeof(T) });
            result = default(T);
            return false;
        }
        #endregion
        
        public bool ApplyUnaryOp_float32(Type c, NdArray<float> in1, NdArray<float> @out)
        {
            // Special handling of the copy operator as it happens "in-place" (kind-of)
            if (typeof(NumCIL.Generic.Operators.ICopyOperation).IsAssignableFrom(c))
            {
                using (var v1 = new PInvoke.bh_multi_array_float32_p(@in1))
                using (var v2 = new PInvoke.bh_multi_array_float32_p(@out))
                    PInvoke.bh_multi_array_float32_assign_array(v2, v1);
                
                if (@out.DataAccessor is DataAccessor_float32)
                    ((DataAccessor_float32)@out.DataAccessor).SetDirty();
                else
                {
                    Utility.Flush();
                    PinnedArrayTracker.Release();
                }
                
                return true;
            }
        
            Func<multi32, multi32> m;
            
            // This lookup prevents a linear scan of the supported operands
            if (!m_unOpLookup.TryGetValue(c, out m))
            {
                m = (from n in m_unOps
                                 where n.Item1.IsAssignableFrom(c)
                                 select n.Item2).FirstOrDefault();
                m_unOpLookup[c] = m;
            }
            
            if (m == null)
            {
                Console.WriteLine("No registered match for: {0}", c.FullName);
                return false;
            }
            
            /*if (@in1.DataAccessor.Length == 1 && @in1.DataAccessor.GetType() == typeof(DefaultAccessor<float>))
                m = PInvoke.float32_*/
        
            using (var v1 = new PInvoke.bh_multi_array_float32_p(@in1))
            {
                Console.WriteLine("Applying op {0}!", c.FullName);
            
                //using (var r = m(v1))
                var r = m(v1);
                {
                    //r.Temp = false;
                    
                    //if (!(@out.DataAccessor is DataAccessor_float32) || ((DataAccessor_float32)@out.DataAccessor).IsAllocated || ((DataAccessor_float32)@out.DataAccessor).MultiArray.IsAllocated)
                    {
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(@out))
                        {
                            Console.WriteLine("Before assign");
                            PInvoke.bh_multi_array_float32_assign_array(v0, r);
                            Console.WriteLine("Disposing v0");
                        }
                    }
                    /*else
                    {                
                        Console.WriteLine("Applied add to un-allocated target");
                        var m = PInvoke.bh_multi_array_float32_new_from_base(r.Base);
                        m.Temp = false;
                        d0.MultiArray = m;
                        // Do we want partial unlink support?
                        // We need the temp array to be un-attached
                        // from the base, but keep the meta.base
                        r.Unlink();
                    }*/
                    
                    if (@out.DataAccessor is DataAccessor_float32)
                        ((DataAccessor_float32)@out.DataAccessor).SetDirty();
                    else
                    {
                        Utility.Flush();
                        PinnedArrayTracker.Release();
                    }
                    Console.WriteLine("Disposing r");
                }
                    
                Console.WriteLine("Apply'ed op {0}!", c.FullName);

                return true;
            }
        }

        
        public bool ApplyBinaryOp_float32(Type c, NdArray<float> in1, NdArray<float> in2, NdArray<float> @out)
        {
            Func<multi32, multi32, multi32> m;
            
            // This lookup prevents a linear scan of the supported operands
            if (!m_binOpLookup.TryGetValue(c, out m))
            {
                m = (from n in m_binOps
                                 where n.Item1.IsAssignableFrom(c)
                                 select n.Item2).FirstOrDefault();
                m_binOpLookup[c] = m;
            }
            
            if (m == null)
            {
                Console.WriteLine("No registered match for: {0}", c.FullName);
                return false;
            }
            
                
            // If the accessor is CIL-managed, we register a GC handle for the array
            // If the input is used, no special action is performed until 
            // a sync is executed, then all the BH queue is flushed and 
            // the GC handles released
            
            // If the output is CIL-managed, we must sync immediately
            
            Console.WriteLine("Preparing operands");
            
            using (var v1 = new PInvoke.bh_multi_array_float32_p(@in1))
            using (var v2 = new PInvoke.bh_multi_array_float32_p(@in2))
            {
                Console.WriteLine("Applying op {0}!", c.FullName);
            
                //using (var r = m(v1, v2))
                var r = m(v1, v2);
                {
                    //r.Temp = false;
                    
                    //if (!(@out.DataAccessor is DataAccessor_float32) || ((DataAccessor_float32)@out.DataAccessor).IsAllocated || ((DataAccessor_float32)@out.DataAccessor).MultiArray.IsAllocated)
                    {
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(@out))
                        {
                            PInvoke.bh_multi_array_float32_assign_array(v0, r);
                            Console.WriteLine("Disposing v0");
                        }
                    }
                    /*else
                    {                
                        Console.WriteLine("Applied add to un-allocated target");
                        var m = PInvoke.bh_multi_array_float32_new_from_base(r.Base);
                        m.Temp = false;
                        d0.MultiArray = m;
                        // Do we want partial unlink support?
                        // We need the temp array to be un-attached
                        // from the base, but keep the meta.base
                        r.Unlink();
                    }*/
                    
                    if (@out.DataAccessor is DataAccessor_float32)
                        ((DataAccessor_float32)@out.DataAccessor).SetDirty();
                    else
                    {
                        Utility.Flush();
                        PinnedArrayTracker.Release();
                    }
                    Console.WriteLine("Disposing r");
                }
                    
                Console.WriteLine("Apply'ed op {0}!", c.FullName);

                return true;
            }
        }          
    }    
}

