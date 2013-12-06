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

        private Tuple<Type, Func<float, multi32, multi32>>[] m_binOpsScalarLhs = 
        {
            new Tuple<Type, Func<float, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IAdd), PInvoke.bh_multi_array_float32_add_scalar_lhs),
            new Tuple<Type, Func<float, multi32, multi32>>(typeof(NumCIL.Generic.Operators.ISub), PInvoke.bh_multi_array_float32_subtract_scalar_lhs),
            new Tuple<Type, Func<float, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMul), PInvoke.bh_multi_array_float32_multiply_scalar_lhs),
            new Tuple<Type, Func<float, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IDiv), PInvoke.bh_multi_array_float32_divide_scalar_lhs),
            new Tuple<Type, Func<float, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMod), PInvoke.bh_multi_array_float32_modulo_scalar_lhs),
            new Tuple<Type, Func<float, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMax), PInvoke.bh_multi_array_float32_maximum_scalar_lhs),
            new Tuple<Type, Func<float, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMin), PInvoke.bh_multi_array_float32_minimin_scalar_lhs),
        };

        private Tuple<Type, Func<multi32, float, multi32>>[] m_binOpsScalarRhs = 
        {
            new Tuple<Type, Func<multi32, float, multi32>>(typeof(NumCIL.Generic.Operators.IAdd), PInvoke.bh_multi_array_float32_add_scalar_rhs),
            new Tuple<Type, Func<multi32, float, multi32>>(typeof(NumCIL.Generic.Operators.ISub), PInvoke.bh_multi_array_float32_subtract_scalar_rhs),
            new Tuple<Type, Func<multi32, float, multi32>>(typeof(NumCIL.Generic.Operators.IMul), PInvoke.bh_multi_array_float32_multiply_scalar_rhs),
            new Tuple<Type, Func<multi32, float, multi32>>(typeof(NumCIL.Generic.Operators.IDiv), PInvoke.bh_multi_array_float32_divide_scalar_rhs),
            new Tuple<Type, Func<multi32, float, multi32>>(typeof(NumCIL.Generic.Operators.IMod), PInvoke.bh_multi_array_float32_modulo_scalar_rhs),
            new Tuple<Type, Func<multi32, float, multi32>>(typeof(NumCIL.Generic.Operators.IMax), PInvoke.bh_multi_array_float32_maximum_scalar_rhs),
            new Tuple<Type, Func<multi32, float, multi32>>(typeof(NumCIL.Generic.Operators.IMin), PInvoke.bh_multi_array_float32_minimin_scalar_rhs),
        };
        
        private Tuple<Type, Action<multi32, multi32>>[] m_binOpsInPlace = 
        {
            new Tuple<Type, Action<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IAdd), PInvoke.bh_multi_array_float32_add_in_place),
            new Tuple<Type, Action<multi32, multi32>>(typeof(NumCIL.Generic.Operators.ISub), PInvoke.bh_multi_array_float32_subtract_in_place),
            new Tuple<Type, Action<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMul), PInvoke.bh_multi_array_float32_multiply_in_place),
            new Tuple<Type, Action<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IDiv), PInvoke.bh_multi_array_float32_divide_in_place),
            new Tuple<Type, Action<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMod), PInvoke.bh_multi_array_float32_modulo_in_place),
            //new Tuple<Type, Action<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMax), PInvoke.bh_multi_array_float32_maximum_in_place),
            //new Tuple<Type, Action<multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMin), PInvoke.bh_multi_array_float32_minimin_in_place),
        };
        
        private Tuple<Type, Action<multi32, float>>[] m_binOpsInPlaceScalarRhs = 
        {
            new Tuple<Type, Action<multi32, float>>(typeof(NumCIL.Generic.Operators.IAdd), PInvoke.bh_multi_array_float32_add_in_place_scalar_rhs),
            new Tuple<Type, Action<multi32, float>>(typeof(NumCIL.Generic.Operators.ISub), PInvoke.bh_multi_array_float32_subtract_in_place_scalar_rhs),
            new Tuple<Type, Action<multi32, float>>(typeof(NumCIL.Generic.Operators.IMul), PInvoke.bh_multi_array_float32_multiply_in_place_scalar_rhs),
            new Tuple<Type, Action<multi32, float>>(typeof(NumCIL.Generic.Operators.IDiv), PInvoke.bh_multi_array_float32_divide_in_place_scalar_rhs),
            new Tuple<Type, Action<multi32, float>>(typeof(NumCIL.Generic.Operators.IMod), PInvoke.bh_multi_array_float32_modulo_in_place_scalar_rhs),
            //new Tuple<Type, Action<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMax), PInvoke.bh_multi_array_float32_maximum_in_place_scalar_rhs),
            //new Tuple<Type, Action<multi32, multi32, multi32>>(typeof(NumCIL.Generic.Operators.IMin), PInvoke.bh_multi_array_float32_minimin_in_place_scalar_rhs),
        };
        
        private Dictionary<Type, Func<multi32, multi32, multi32>> m_binOpLookup = new Dictionary<Type, Func<multi32, multi32, multi32>>();
        private Dictionary<Type, Func<float, multi32, multi32>> m_binOpScalarLhsLookup = new Dictionary<Type, Func<float, multi32, multi32>>();
        private Dictionary<Type, Func<multi32, float, multi32>> m_binOpScalarRhsLookup = new Dictionary<Type, Func<multi32, float, multi32>>();
        private Dictionary<Type, Func<multi32, multi32>> m_unOpLookup = new Dictionary<Type, Func<multi32, multi32>>();
        private Dictionary<Type, Action<multi32, float>> m_binOpInPlaceScalarRhsLookup = new Dictionary<Type, Action<multi32, float>>();
        private Dictionary<Type, Action<multi32, multi32>> m_binOpInPlaceLookup = new Dictionary<Type, Action<multi32, multi32>>();

        #region ITypedApplyImplementor implementation

        public bool ApplyBinaryOp(Type c, NdArray<float> in1, NdArray<float> in2, NdArray<float> @out)
        {
            var isScalarIn1 = @in1.DataAccessor.GetType() == typeof(DefaultAccessor<float>) && @in1.DataAccessor.Length == 1;
            var isScalarIn2 = @in2.DataAccessor.GetType() == typeof(DefaultAccessor<float>) && @in2.DataAccessor.Length == 1;
            
            Action<NdArray<float>, NdArray<float>, NdArray<float>> exec = null;
            
            if (exec == null && isScalarIn2 && in1.DataAccessor == @out.DataAccessor && @out.DataAccessor is Bohrium2.DataAccessor_float32)
            {
                Action<multi32, float> minplacerhs;
                
                if (!m_binOpInPlaceScalarRhsLookup.TryGetValue(c, out minplacerhs))
                {
                    minplacerhs = (from n in m_binOpsInPlaceScalarRhs
                                                  where n.Item1.IsAssignableFrom(c)
                                                  select n.Item2).FirstOrDefault();
                     
                    m_binOpInPlaceScalarRhsLookup[c] = minplacerhs;
                }
                
                if (minplacerhs != null)
                    exec = (_in1, _in2, _out) => {
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(_out))
                            minplacerhs(v0, in2.DataAccessor[0]);
                    };
            }

            if (exec == null && in1.DataAccessor == @out.DataAccessor && @out.DataAccessor is Bohrium2.DataAccessor_float32)
            {
                Action<multi32, multi32> minplace;
                
                if (!m_binOpInPlaceLookup.TryGetValue(c, out minplace))
                {
                    minplace = (from n in m_binOpsInPlace
                                                  where n.Item1.IsAssignableFrom(c)
                                                  select n.Item2).FirstOrDefault();
                     
                    m_binOpInPlaceLookup[c] = minplace;
                }
                
                if (minplace != null)
                    exec = (_in1, _in2, _out) => {
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(_out))
                        using (var v2 = new PInvoke.bh_multi_array_float32_p(_in2))
                            minplace(v0, v2);
                    };
            }
            

            // Not sure this extra code is worth the trouble for dealing with scalars
            if (exec == null && isScalarIn1)
            {
                Func<float, multi32, multi32> mlhs;
                
                if (!m_binOpScalarLhsLookup.TryGetValue(c, out mlhs))
                {
                    mlhs = (from n in m_binOpsScalarLhs
                                           where n.Item1.IsAssignableFrom(c)
                                           select n.Item2).FirstOrDefault();
                     
                    m_binOpScalarLhsLookup[c] = mlhs;
                }
                
                if (mlhs != null)
                {
                    exec = (_in1, _in2, _out) =>
                    {
                        using (var v2 = new PInvoke.bh_multi_array_float32_p(_in2))
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(_out))
                            PInvoke.bh_multi_array_float32_assign_array(v0, mlhs(_in1.DataAccessor[0], v2));
                    };
                }
            }

            if (isScalarIn2 && exec == null)
            {
                Func<multi32, float, multi32> mrhs;
                
                if (!m_binOpScalarRhsLookup.TryGetValue(c, out mrhs))
                {
                    mrhs = (from n in m_binOpsScalarRhs
                                           where n.Item1.IsAssignableFrom(c)
                                           select n.Item2).FirstOrDefault();
                     
                    m_binOpScalarRhsLookup[c] = mrhs;
                }
                
                if (mrhs != null)
                {
                    exec = (_in1, _in2, _out) =>
                    {
                        using (var v1 = new PInvoke.bh_multi_array_float32_p(_in1))
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(_out))
                            PInvoke.bh_multi_array_float32_assign_array(v0, mrhs(v1, _in2.DataAccessor[0]));
                    };

                }
            }
            
            if (exec == null)
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
            
                if (m != null)
                {                
                    exec = (_in1, _in2, _out) =>
                    {
                        using (var v1 = new PInvoke.bh_multi_array_float32_p(@in1))
                        using (var v2 = new PInvoke.bh_multi_array_float32_p(@in2))
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(@out))
                            PInvoke.bh_multi_array_float32_assign_array(v0, m(v1, v2));
                    };  
                }              
            }

            if (exec == null)
            {
                Console.WriteLine("No registered match for: {0}", c.FullName);
                return false;
            }
                
            // If the accessor is CIL-managed, we register a GC handle for the array
            // If the input is used, no special action is performed until 
            // a sync is executed, then all the BH queue is flushed and 
            // the GC handles released
            
            // If the output is CIL-managed, we must sync immediately
            
            exec(@in1, @in2, @out);            
            if (@out.DataAccessor is DataAccessor_float32)
                ((DataAccessor_float32)@out.DataAccessor).SetDirty();
            else
            {
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
                            PInvoke.bh_multi_array_float32_assign_scalar(v2, _in.DataAccessor[0]);
                    else
                        using (var v1 = new PInvoke.bh_multi_array_float32_p(_in))
                        using (var v2 = new PInvoke.bh_multi_array_float32_p(_out))
                            PInvoke.bh_multi_array_float32_assign_array(v2, v1);
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
                            PInvoke.bh_multi_array_float32_assign_array(v0, m(v1));
                    };
                }
            }
            
            if (exec == null)
            {
                Console.WriteLine("No registered match for: {0}", c.FullName);
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

