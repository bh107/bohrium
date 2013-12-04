using System;
using System.Linq;
using System.Collections.Generic;
using NumCIL.Generic;

using multi32 = NumCIL.Bohrium2.PInvoke.bh_multi_array_float32_p;

namespace NumCIL.Bohrium2
{    

/// <summary>
    /// Basic factory for creating Bohrium accessors
    /// </summary>
    /// <typeparam name="T">The type of data kept in the underlying array</typeparam>
    public class BohriumAccessorFactory<T> : NumCIL.Generic.IAccessorFactory<T>
    {
        /// <summary>
        /// Creates a new accessor for a data chunk of the given size
        /// </summary>
        /// <param name="size">The size of the array</param>
        /// <returns>An accessor</returns>
        public IDataAccessor<T> Create(long size) 
        { 
            if (typeof(T) == typeof(float))
                return size == 1 ? (IDataAccessor<T>)new DefaultAccessor<T>(new T[1]) : (IDataAccessor<T>)new DataAccessor_float32(size); 
                
            return new DefaultAccessor<T>(size);
        }
        /// <summary>
        /// Creates a new accessor for a preallocated array
        /// </summary>
        /// <param name="data">The data to wrap</param>
        /// <returns>An accessor</returns>
        public IDataAccessor<T> Create(T[] data)
        {                
            if (data.Length == 1)
                return new DefaultAccessor<T>(data);
                
            if (typeof(T) == typeof(float))
                return (IDataAccessor<T>)new DataAccessor_float32((float[])(object)data); 
            
            return new DefaultAccessor<T>(data);
        }
    }

    public class ApplyImplementor : UFunc.IApplyBinaryOp, UFunc.IApplyUnaryOp
    {
        public ApplyImplementor()
        {
            NumCIL.UFunc.ApplyManager.DEBUG_FALLBACK = (a, b) => Console.WriteLine("*** Unhandled op {0} for types [ {1} ]", a.FullName,string.Join(",", b.Select(n => n.ToString())));
        }

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
    
        #region IApplyBinaryOp implementation
        public bool ApplyBinaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> in2, NdArray<T> @out) 
            where C : struct, IBinaryOp<T>
        {
            if (typeof(T) == typeof(float))
                return ApplyBinaryOp_float32(typeof(C), (NdArray<float>)(object)in1, (NdArray<float>)(object)in2, (NdArray<float>)(object)@out);

            return false;
        }
        #endregion

        #region IApplyUnaryOp implementation

        public bool ApplyUnaryOp<T, C>(C op, NdArray<T> in1, NdArray<T> @out) where C : struct, IUnaryOp<T>
        {
            if (typeof(T) == typeof(float))
                return ApplyUnaryOp_float32(typeof(C), (NdArray<float>)(object)in1, (NdArray<float>)(object)@out);
            
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
                
                return true;
            }
            
            var m = (from n in m_unOps
                              where n.Item1.IsAssignableFrom(c)
                              select n.Item2).FirstOrDefault();
            if (m == null)
            {
                Console.WriteLine("No registered match for: {0}", c.FullName);
                return false;
            }
        
            var d0 = (DataAccessor_float32)@out.DataAccessor;
            using (var v1 = new PInvoke.bh_multi_array_float32_p(@in1))
            {
                Console.WriteLine("Applying op {0}!", c.FullName);
            
                using (var r = m(v1))
                {
                    r.Temp = false;
                    
                    //if (d0.IsAllocated || d0.MultiArray.IsAllocated)
                    {
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(@out))
                            PInvoke.bh_multi_array_float32_assign_array(v0, r);
                    }
                    /*else
                    {                
                        var m = PInvoke.bh_multi_array_float32_new_from_base(r.Base);
                        m.Temp = false;
                        d0.MultiArray = m;
                        r.Unlink();
                    }*/
                    
                    d0.SetDirty();
                }
                    
                Console.WriteLine("Apply'ed op {0}!", c.FullName);

                return true;
            }
        }

        
        public bool ApplyBinaryOp_float32(Type c, NdArray<float> in1, NdArray<float> in2, NdArray<float> @out)
        {
            var m = (from n in m_binOps
                              where n.Item1.IsAssignableFrom(c)
                              select n.Item2).FirstOrDefault();
            
            if (m == null)
            {
                Console.WriteLine("No registered match for: {0}", c.FullName);
                return false;
            }
        
            var d0 = (DataAccessor_float32)@out.DataAccessor;
            using (var v1 = new PInvoke.bh_multi_array_float32_p(@in1))
            using (var v2 = new PInvoke.bh_multi_array_float32_p(@in2))
            {
                Console.WriteLine("Applying op {0}!", c.FullName);
            
                using (var r = m(v1, v2))
                {
                    r.Temp = false;
                    
                    //if (d0.IsAllocated || d0.MultiArray.IsAllocated)
                    {
                        using (var v0 = new PInvoke.bh_multi_array_float32_p(@out))
                            PInvoke.bh_multi_array_float32_assign_array(v0, r);
                    }
                    /*else
                    {                
                        Console.WriteLine("Applied add to un-allocated target");
                        var m = PInvoke.bh_multi_array_float32_new_from_base(r.Base);
                        m.Temp = false;
                        d0.MultiArray = m;
                        r.Unlink();
                    }*/
                    
                    d0.SetDirty();
                }
                    
                Console.WriteLine("Apply'ed op {0}!", c.FullName);

                return true;
            }
        }
          
    }    
}

