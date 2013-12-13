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
                Console.WriteLine("No registered match for: {0}: {1}", c.FullName, @out.DataAccessor.GetType().FullName, in1.DataAccessor.GetType().FullName, in2.DataAccessor.GetType().FullName);
                return false;
            }
                
            // If the accessor is CIL-managed, we register a GC handle for the array
            // If the input is used, no special action is performed until 
            // a sync is executed, then all the BH queue is flushed and 
            // the GC handles released
                        
            using (var v1 = new PInvoke.bh_multi_array_float32_p(@in1))
            using (var v2 = new PInvoke.bh_multi_array_float32_p(@in2))
            using (var v0 = new PInvoke.bh_multi_array_float32_p(@out))
            {
                PInvoke.bh_multi_array_float32_assign_array(v0, m(v1, v2));
                if (!(@out.DataAccessor is DataAccessor_float32))
                    v0.Sync();
            }

            if (@out.DataAccessor is DataAccessor_float32)
                ((DataAccessor_float32)@out.DataAccessor).SetDirty();
            else
            {
                // If the output is CIL-managed, we must sync immediately
                Utility.Flush();
                PinnedArrayTracker.Release();
            }
            
            return true;
        }

        public bool ApplyBinaryConvOp<Ta>(Type c, NdArray<Ta> in1, NdArray<Ta> in2, NdArray<float> @out)
        {
            return false;
        }

        public bool ApplyUnaryOp(Type c, NdArray<float> in1, NdArray<float> @out)
        {
            var isScalarIn1 = @in1.DataAccessor.GetType() == typeof(DefaultAccessor<float>) && @in1.DataAccessor.Length == 1;
            
            Action<NdArray<float>, NdArray<float>> exec = null;
            
            // Special handling of the copy operator as it happens "in-place" (kind-of)
            if (typeof(NumCIL.Generic.Operators.ICopyOperation).IsAssignableFrom(c))
            {
                exec = (_in, _out) =>
                {
                    if (isScalarIn1)
                        using (var v2 = new PInvoke.bh_multi_array_float32_p(_out))
                        {
                            PInvoke.bh_multi_array_float32_assign_scalar(v2, _in.DataAccessor[0]);
                            if (!(_out.DataAccessor is DataAccessor_float32))
                                v2.Sync();
                    }
                    else
                        using (var v1 = new PInvoke.bh_multi_array_float32_p(_in))
                        using (var v2 = new PInvoke.bh_multi_array_float32_p(_out))
                        {
                            PInvoke.bh_multi_array_float32_assign_array(v2, v1);
                            if (!(_out.DataAccessor is DataAccessor_float32))
                                v2.Sync();
                        }
                };
            }
                    
            if (exec != null)
            {
                Func<multi32, multi32> m;
            
                // This lookup prevents a linear scan of the supported operands
                if (!m_unOpLookup.TryGetValue(c, out m))
                {
                    m = (from n in m_unOps
                            where n.Item1.IsAssignableFrom(c)
                            select n.Item2).FirstOrDefault();
                    m_unOpLookup[c] = m;
                }
                
                if (m != null)
                {
                    exec = (_in, _out) =>
                    {
                        using (var v1 = new PInvoke.bh_multi_array_float32_p(_in))
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(_out))
                        {
                            PInvoke.bh_multi_array_float32_assign_array(v0, m(v1));
                            if (!(_out.DataAccessor is DataAccessor_float32))
                                v0.Sync();
                        }
                    };
                }
            }
            
            if (exec == null)
            {
                Console.WriteLine("No registered match for: {0}: ", c.FullName, @out.DataAccessor.GetType().FullName, in1.DataAccessor.GetType().FullName);
                return false;
            }
            
            exec(@in1, @out);
            
            if (@out.DataAccessor is DataAccessor_float32)
                ((DataAccessor_float32)@out.DataAccessor).SetDirty();
            else
            {
                Utility.Flush();
                PinnedArrayTracker.Release();
            }
            
            return true;
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

