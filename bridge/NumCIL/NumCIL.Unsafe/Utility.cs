#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;
using System.Reflection;

namespace NumCIL.Unsafe
{
    /// <summary>
    /// Utility functions for unsafe methods
    /// </summary>
    public static class Utility
    {
        /// <summary>
        /// Gets a value describing if unsafe methods are supported
        /// </summary>
        public static bool SupportsUnsafe
        {
            get
            {
                try
                {
                    return DoTest() == 1;
                }
                catch
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the apply operation, bound to the data type, but unbound in the operand type.
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, but unbound in the operand type, or null if no such method exists</returns>
        public static MethodInfo GetBinaryApply<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "UFunc_Op_Inner_Binary_Flush_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.Apply).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }

		/// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the apply operation, bound to the data type, but unbound in the operand type.
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, but unbound in the operand type, or null if no such method exists</returns>
        public static MethodInfo GetBinaryLhsScalarApply<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "UFunc_Op_Inner_Binary_LhsScalar_Flush_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.Apply).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }

		/// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the apply operation, bound to the data type, but unbound in the operand type.
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, but unbound in the operand type, or null if no such method exists</returns>
        public static MethodInfo GetBinaryRhsScalarApply<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "UFunc_Op_Inner_Binary_RhsScalar_Flush_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.Apply).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }

        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the apply operation, bound to the data type, but unbound in the operand type.
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, but unbound in the operand type, or null if no such method exists</returns>
        public static MethodInfo GetUnaryApply<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "UFunc_Op_Inner_Unary_Flush_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.Apply).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }

		/// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the apply operation, bound to the data type, but unbound in the operand type.
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, but unbound in the operand type, or null if no such method exists</returns>
        public static MethodInfo GetUnaryScalarApply<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "UFunc_Op_Inner_Unary_Scalar_Flush_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.Apply).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }

        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the apply operation, bound to the data type, but unbound in the operand type.
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, but unbound in the operand type, or null if no such method exists</returns>
        public static MethodInfo GetNullaryApply<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "UFunc_Op_Inner_Nullary_Flush_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.Apply).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }

        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the aggregate operation, bound to the data type, but unbound in the operand type.
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, but unbound in the operand type, or null if no such method exists</returns>
        public static MethodInfo GetAggregate<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "Aggregate_Entry_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.Aggregate).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }

        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the reduce operation, bound to the data type, but unbound in the operand type.
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, but unbound in the operand type, or null if no such method exists</returns>
        public static MethodInfo GetReduce<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "UFunc_Reduce_Inner_Flush_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.Reduce).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }

        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the copy operation, bound for the data type
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, or null if no such method exists</returns>
        public static MethodInfo GetCopyToManaged<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            return typeof(NumCIL.Unsafe.Copy).GetMethod("Memcpy", new Type[] { typeof(T[]), typeof(IntPtr), typeof(long) });
        }

        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for the copy operation, bound for the data type
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, or null if no such method exists</returns>
        public static MethodInfo GetCopyFromManaged<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            return typeof(NumCIL.Unsafe.Copy).GetMethod("Memcpy", new Type[] { typeof(IntPtr), typeof(T[]), typeof(long) });
        }

        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for creating an accessor based on size, bound for the data type
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, or null if no such method exists</returns>
        public static MethodInfo GetCreateAccessorSize<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "CreateFromSize_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.CreateAccessor).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }


        /// <summary>
        /// Gets a <see cref="System.Reflection.MethodInfo"/> instance for creating an accessor based on an array, bound for the data type
        /// Returns null if the data type is not supported.
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <returns>A <see cref="System.Reflection.MethodInfo"/> instance, bound to the data type, or null if no such method exists</returns>
        public static MethodInfo GetCreateAccessorData<T>()
        {
            if (!typeof(T).IsPrimitive || !SupportsUnsafe)
                return null;

            string name = "CreateFromData_" + typeof(T).Name.Replace(".", "_");
            return typeof(NumCIL.Unsafe.CreateAccessor).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        }
        
        /// <summary>
        /// Method that performs an unsafe operation
        /// </summary>
        /// <returns>A test value</returns>
        private static long DoTest()
        {
            long[] x = new long[1];
            unsafe
            {
                fixed (long* y = x)
                {
                    y[0] = 1;
                }
            }

            return x[0];
        }
    }
}
