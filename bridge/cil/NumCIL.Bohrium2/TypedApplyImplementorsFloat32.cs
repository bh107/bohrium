using System;
using NumCIL.Generic;
using System.Collections.Generic;
using System.Linq;

using ma_float32 = NumCIL.Bohrium2.PInvoke.bh_multi_array_float32_p;
using ma_bool8 = NumCIL.Bohrium2.PInvoke.bh_multi_array_bool8_p;
using float32 = System.Single;

namespace NumCIL.Bohrium2
{
    public class ApplyImplementor_float32 : ITypedApplyImplementor<float>
    {
        private Tuple<Type, Func<ma_float32, ma_float32>>[] m_unOps = 
        {
            new Tuple<Type, Func<ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IAbs), PInvoke.bh_multi_array_float32_absolute),
            new Tuple<Type, Func<ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IFloor), PInvoke.bh_multi_array_float32_floor),
            new Tuple<Type, Func<ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.ICeiling), PInvoke.bh_multi_array_float32_ceil),
            new Tuple<Type, Func<ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IRound), PInvoke.bh_multi_array_float32_rint),
            new Tuple<Type, Func<ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.ISqrt), PInvoke.bh_multi_array_float32_sqrt),
            new Tuple<Type, Func<ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IExp), PInvoke.bh_multi_array_float32_exp),
            new Tuple<Type, Func<ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IAbs), PInvoke.bh_multi_array_float32_absolute)
        };
        
        private Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>[] m_binOps = 
        {
            new Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IAdd), PInvoke.bh_multi_array_float32_add),
            new Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.ISub), PInvoke.bh_multi_array_float32_subtract),
            new Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IMul), PInvoke.bh_multi_array_float32_multiply),
            new Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IDiv), PInvoke.bh_multi_array_float32_divide),
            new Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IMod), PInvoke.bh_multi_array_float32_modulo),
            new Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IMax), PInvoke.bh_multi_array_float32_maximum),
            new Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IMin), PInvoke.bh_multi_array_float32_minimin),
            new Tuple<Type, Func<ma_float32, ma_float32, ma_float32>>(typeof(NumCIL.Generic.Operators.IPow), PInvoke.bh_multi_array_float32_power),
        };
        
        private Tuple<Type, Func<ma_float32, float32>>[] m_aggOps = 
        {
            new Tuple<Type, Func<ma_float32, float32>>(typeof(NumCIL.Generic.Operators.IAdd), PInvoke.bh_multi_array_float32_sum),
            new Tuple<Type, Func<ma_float32, float32>>(typeof(NumCIL.Generic.Operators.IMul), PInvoke.bh_multi_array_float32_product),
            new Tuple<Type, Func<ma_float32, float32>>(typeof(NumCIL.Generic.Operators.IMin), PInvoke.bh_multi_array_float32_min),
            new Tuple<Type, Func<ma_float32, float32>>(typeof(NumCIL.Generic.Operators.IMax), PInvoke.bh_multi_array_float32_max)
        };
        
        private Tuple<Type, Func<ma_float32, long, ma_float32>>[] m_reduceOps = 
        {
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IAdd), PInvoke.bh_multi_array_float32_partial_reduce_add),
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IMul), PInvoke.bh_multi_array_float32_partial_reduce_multiply),
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IMin), PInvoke.bh_multi_array_float32_partial_reduce_min),
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IMax), PInvoke.bh_multi_array_float32_partial_reduce_max),
            
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IAnd), PInvoke.bh_multi_array_float32_partial_reduce_bitwise_and),
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IOr), PInvoke.bh_multi_array_float32_partial_reduce_bitwise_or),
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IXor), PInvoke.bh_multi_array_float32_partial_reduce_bitwise_xor),
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IAnd), PInvoke.bh_multi_array_float32_partial_reduce_logical_and),
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IOr), PInvoke.bh_multi_array_float32_partial_reduce_logical_or),
            new Tuple<Type, Func<ma_float32, long, ma_float32>>(typeof(NumCIL.Generic.Operators.IXor), PInvoke.bh_multi_array_float32_partial_reduce_logical_xor),
        };
        
        private Dictionary<Type, Func<ma_float32, ma_float32, ma_float32>> m_binOpLookup = new Dictionary<Type, Func<ma_float32, ma_float32, ma_float32>>();
        private Dictionary<Type, Func<ma_float32, ma_float32>> m_unOpLookup = new Dictionary<Type, Func<ma_float32, ma_float32>>();
        private Dictionary<Type, Func<ma_float32, float32>> m_aggLookup = new Dictionary<Type, Func<ma_float32, float>>();
        private Dictionary<Type, Func<ma_float32, long, ma_float32>> m_reduceLookup = new Dictionary<Type, Func<ma_float32, long, ma_float32>>();

        #region ITypedApplyImplementor implementation

        public bool ApplyBinaryOp(Type c, NdArray<float> in1, NdArray<float> in2, NdArray<float> @out)
        {
            Func<ma_float32, ma_float32, ma_float32> m;
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
                Console.WriteLine("No registered match for: {0}: {1}, {2}, {3}", c.FullName, @out.DataAccessor.GetType().FullName, in1.DataAccessor.GetType().FullName, in2.DataAccessor.GetType().FullName);
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
                    
            if (exec == null)
            {
                Func<ma_float32, ma_float32> m;
            
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
                Console.WriteLine("No registered match for: {0}: {1}, {2}", c.FullName, @out.DataAccessor.GetType().FullName, in1.DataAccessor.GetType().FullName);
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

        private bool DoConvert<T>(NdArray<float> @out, Func<T> constructor, Func<T, ma_float32> converter)
            where T : IDisposable
        {
            using (var v0 = new PInvoke.bh_multi_array_float32_p(@out))
            using (var v1 = constructor())
            {
                PInvoke.bh_multi_array_float32_assign_array(v0, converter(v1));
                if (!(@out.DataAccessor is DataAccessor_float32))
                    v0.Sync();
            }
            
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
            if (typeof(NumCIL.Generic.Operators.ITypeConversion).IsAssignableFrom(c))
            {
                if (typeof(Ta) == typeof(float))
                    return ApplyUnaryOp(c, (NdArray<float>)(object)in1, @out);
                else if (typeof(Ta) == typeof(double))
                    return DoConvert(@out, 
                        () => new PInvoke.bh_multi_array_float64_p((NdArray<double>)(object)in1),
                        PInvoke.bh_multi_array_float32_convert_float64);
                else if (typeof(Ta) == typeof(long))
                    return DoConvert(@out, 
                        () => new PInvoke.bh_multi_array_int64_p((NdArray<long>)(object)in1),
                        PInvoke.bh_multi_array_float32_convert_int64);
            }
            /*else if (typeof(NumCIL.Generic.Operators.IRealValue).IsAssignableFrom(c) && (typeof(Ta) == typeof(NumCIL.Complex64.DataType)))
                return DoConvert(@out, 
                    () => new PInvoke.bh_multi_array_complex64_p((NdArray<NumCIL.Complex64.DataType>)(object)in1),
                    PInvoke.bh_multi_array_float64_convert_complex64_real);
            else if (typeof(NumCIL.Generic.Operators.IImaginaryValue).IsAssignableFrom(c) && (typeof(Ta) == typeof(NumCIL.Complex64.DataType)))
                return DoConvert(@out, 
                    () => new PInvoke.bh_multi_array_complex64_p((NdArray<NumCIL.Complex64.DataType>)(object)in1),
                    PInvoke.bh_multi_array_float64_convert_complex64_imag);
            */
            return false;
        }

        public bool ApplyNullaryOp(Type c, NdArray<float> @out)
        {
            return false;
        }

        public bool ApplyReduce(Type c, long axis, NdArray<float> in1, NdArray<float> @out)
        {
            Func<ma_float32, long, ma_float32> m;
            if (!m_reduceLookup.TryGetValue(c, out m))
            {
                m = (from n in m_reduceOps
                                 where n.Item1.IsAssignableFrom(c)
                                 select n.Item2).FirstOrDefault();
                m_reduceLookup[c] = m;
            }
            
            if (m == null)
            {
                Console.WriteLine("No registered match for reduce: {0}: {1}", c.FullName, in1.DataAccessor.GetType().FullName);
                return false;
            }
            
            using (var v1 = new PInvoke.bh_multi_array_float32_p(@in1))
            using (var v0 = new PInvoke.bh_multi_array_float32_p(@out))
            {
                PInvoke.bh_multi_array_float32_assign_array(v0, m(v1, axis));
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

        public bool ApplyMatmul(Type cadd, Type cmul, NdArray<float> in1, NdArray<float> in2, NdArray<float> @out = null)
        {
            return false;
        }

        public bool ApplyAggregate(Type c, NdArray<float> in1, out float result)
        {
            Func<ma_float32, float32> m;
            if (!m_aggLookup.TryGetValue(c, out m))
            {
                m = (from n in m_aggOps
                                 where n.Item1.IsAssignableFrom(c)
                                 select n.Item2).FirstOrDefault();
                m_aggLookup[c] = m;
            }
            
            if (m == null)
            {
                //TODO: Attempt to build one using multiple partial reductions ...
            }
             
            if (m == null)
            {
                Console.WriteLine("No registered match for aggregate: {0}: {1}", c.FullName, in1.DataAccessor.GetType().FullName);
                result = default(float);
                return false;
            }

            using(var v0 = new PInvoke.bh_multi_array_float32_p(in1))
                result = m(v0);

            return true;
        }

        #endregion
    }       
}

