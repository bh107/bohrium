using System;
using NumCIL.Generic;

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
}

