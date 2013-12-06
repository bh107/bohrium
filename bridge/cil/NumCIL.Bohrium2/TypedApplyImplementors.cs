using System;
using NumCIL.Generic;
using System.Collections.Generic;
using System.Linq;

using multi32 = NumCIL.Bohrium2.PInvoke.bh_multi_array_float32_p;

namespace NumCIL.Bohrium2
{
    public class ApplyImplementor_float32 : ITypedApplyImplementor<float>
    {
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

        #region ITypedApplyImplementor implementation

        public bool ApplyBinaryOp(Type c, NdArray<float> in1, NdArray<float> in2, NdArray<float> @out)
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

        public bool ApplyBinaryConvOp<Ta>(Type c, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<float> @out)
        {
            return false;
        }

        public bool ApplyUnaryOp(Type c, NdArray<float> in1, NdArray<float> @out)
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

        public bool ApplyUnaryConvOp<Ta>(Type c, NdArray<Ta> in1, NdArray<float> @out)
        {
            return false;
        }

        public bool ApplyNullaryOp(Type c, NdArray<float> @out)
        {
            return false;
        }

        public bool ApplyReduce(Type c, long axis, NdArray<float> in1, NdArray<float> @out)
        {
            return false;
        }

        public bool ApplyMatmul(Type cadd, Type cmul, NdArray<float> in1, NdArray<float> in2, NdArray<float> @out = null)
        {
            return false;
        }

        public bool ApplyAggregate(Type c, NdArray<float> in1, out float result)
        {
            result = default(float);
            return false;
        }

        #endregion
    }       
}

