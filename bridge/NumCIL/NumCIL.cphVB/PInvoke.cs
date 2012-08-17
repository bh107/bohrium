#region Copyright
/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

//CPHVB basic control types
using cphvb_intp = System.Int64;
using cphvb_index = System.Int64;
using cphvb_type = System.Int64;
using cphvb_enumbase = System.Int32;
using cphvb_data_ptr = System.IntPtr;

//CPHVB Signed data types
using cphvb_bool = System.SByte;
using cphvb_int8 = System.SByte;
using cphvb_int16 = System.Int16;
using cphvb_int32 = System.Int32;
using cphvb_int64 = System.Int64;

//CPHVB Unsigned data types
using cphvb_uint8 = System.Byte;
using cphvb_uint16 = System.UInt16;
using cphvb_uint32 = System.UInt32;
using cphvb_uint64 = System.UInt64;

//CPHVB floating point types
using cphvb_float32 = System.Single;
using cphvb_float64 = System.Double;

//CPHVB complex types
using cphvb_complex64 = NumCIL.Complex64.DataType;
using cphvb_complex128 = System.Numerics.Complex;

namespace NumCIL.cphVB
{
    /// <summary>
    /// Container class for methods and datatypes that call cphVB
    /// </summary>
    public static class PInvoke
    {
        /// <summary>
        /// The statically defined maximum cphVB component name
        /// </summary>
        public const int CPHVB_COMPONENT_NAME_SIZE = 1024;
        /// <summary>
        /// The statically defined maximum number of dimensions
        /// </summary>
        public const int CPHVB_MAXDIM = 16;
        /// <summary>
        /// The statically defined maximum number of operands for built-in cphVB instructions
        /// </summary>
        public const int CPHVB_MAX_NO_OPERANDS = 3;

        /// <summary>
        /// Cached lookup to see if the process is running 64bit
        /// </summary>
        public static readonly bool Is64Bit = IntPtr.Size == 8;
        /// <summary>
        /// The size of an int pointer
        /// </summary>
        public static readonly int INTP_SIZE = Marshal.SizeOf(typeof(cphvb_intp));
        /// <summary>
        /// The size of the largest userfunc struct
        /// </summary>
        public static readonly int USERFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_union));
        /// <summary>
        /// The size of the random userfunc struct
        /// </summary>
        public static readonly int RANDOMFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_random));
        /// <summary>
        /// The size of the reduce userfunc struct
        /// </summary>
        public static readonly int REDUCEFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_reduce));
        /// <summary>
        /// The size of the matmul userfunc struct
        /// </summary>
        public static readonly int MATMULFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_matmul));
        /// <summary>
        /// The size of the plain userfunc struct
        /// </summary>
        public static readonly int PLAINFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_plain));

        /// <summary>
        /// The known component types in cphVB
        /// </summary>
        public enum cphvb_component_type : long
        {
            /// <summary>
            /// The bridge component
            /// </summary>
            CPHVB_BRIDGE,
            /// <summary>
            /// The Virtual Execution Manager component
            /// </summary>
            CPHVB_VEM,
            /// <summary>
            /// The Virtual Execution component
            /// </summary>
            CPHVB_VE,
            /// <summary>
            /// An unknown component type
            /// </summary>
            CPHVB_COMPONENT_ERROR
        }

        /// <summary>
        /// The error codes defined in cphVB
        /// </summary>
        public enum cphvb_error : long
        {
            /// <summary>
            /// General success
            /// </summary>
            CPHVB_SUCCESS,
            /// <summary>
            /// Fatal error
            /// </summary>
            CPHVB_ERROR,
            /// <summary>
            /// Data type not supported
            /// </summary>
            CPHVB_TYPE_NOT_SUPPORTED,
            /// <summary>
            /// Out of memory
            /// </summary>
            CPHVB_OUT_OF_MEMORY,
            /// <summary>
            /// Recoverable
            /// </summary>
            CPHVB_PARTIAL_SUCCESS,
            /// <summary>
            /// Instruction is not executed
            /// </summary>
            CPHVB_INST_PENDING,
            /// <summary>
            /// Instruction not supported
            /// </summary>
            CPHVB_INST_NOT_SUPPORTED,
            /// <summary>
            /// User-defined function not supported
            /// </summary>
            CPHVB_USERFUNC_NOT_SUPPORTED
        }

        /// <summary>
        /// The data types supported by cphVB
        /// </summary>
        public enum cphvb_type : long
        {
            /// <summary>
            /// The boolean datatype
            /// </summary>
            CPHVB_BOOL,
            /// <summary>
            /// The signed 8bit datatype
            /// </summary>
            CPHVB_INT8,
            /// <summary>
            /// The signed 16bit datatype
            /// </summary>
            CPHVB_INT16,
            /// <summary>
            /// The signed 32bit datatype
            /// </summary>
            CPHVB_INT32,
            /// <summary>
            /// The signed 64bit datatype
            /// </summary>
            CPHVB_INT64,
            /// <summary>
            /// The unsigned 8bit datatype
            /// </summary>
            CPHVB_UINT8,
            /// <summary>
            /// The unsigned 16bit datatype
            /// </summary>
            CPHVB_UINT16,
            /// <summary>
            /// The unsigned 32bit datatype
            /// </summary>
            CPHVB_UINT32,
            /// <summary>
            /// The unsigned 64bit datatype
            /// </summary>
            CPHVB_UINT64,
            /// <summary>
            /// The 16bit floating point datatype
            /// </summary>
            CPHVB_FLOAT16,
            /// <summary>
            /// The 32bit floating point datatype
            /// </summary>
            CPHVB_FLOAT32,
            /// <summary>
            /// The 64bit floating point datatype
            /// </summary>
            CPHVB_FLOAT64,
            /// <summary>
            /// The 64bit complex datatype
            /// </summary>
            CPHVB_COMPLEX64,
            /// <summary>
            /// The 128bit complex datatype
            /// </summary>
            CPHVB_COMPLEX128,
            /// <summary>
            /// The unknown datatype
            /// </summary>
            CPHVB_UNKNOWN
        }

        /// <summary>
        /// The configuration dictionary for a component
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        struct dictionary
        {
            /// <summary>
            /// Number of entries in dictionary
            /// </summary>
            public int n;
            /// <summary>
            /// Storage size
            /// </summary>
            public int size;
            /// <summary>
            /// SList of string values
            /// </summary>
            public byte[][] val;
            /// <summary>
            /// List of string keys
            /// </summary>
            public byte[][] key;
            /// <summary>
            /// List of hash values for keys
            /// </summary>
            public uint[] hash;
        }

        /// <summary>
        /// A constant value for a cphVB operation
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct cphvb_constant
        {
            /// <summary>
            /// The value itself
            /// </summary>
            public cphvb_constant_value value;
            /// <summary>
            /// The value type
            /// </summary>
            public cphvb_type type;

            /// <summary>
            /// Constructs a new constant of the specified type
            /// </summary>
            /// <param name="type">The constant type</param>
            /// <param name="v">The constant value</param>
            public cphvb_constant(cphvb_type type, object v)
            {
                this.type = type;
                this.value = new cphvb_constant_value().Set(v);
            }

            /// <summary>
            /// Constructs a new constant using the specified value
            /// </summary>
            /// <param name="v">The constant value</param>
            public cphvb_constant(object v)
            {
                this.value = new cphvb_constant_value().Set(v);
                
                if (v is cphvb_bool)
                    this.type = cphvb_type.CPHVB_BOOL;
                else if (v is cphvb_int16)
                    this.type = cphvb_type.CPHVB_INT16;
                else if (v is cphvb_int32)
                    this.type = cphvb_type.CPHVB_INT32;
                else if (v is cphvb_int64)
                    this.type = cphvb_type.CPHVB_INT64;
                else if (v is cphvb_uint8)
                    this.type = cphvb_type.CPHVB_UINT8;
                else if (v is cphvb_uint16)
                    this.type = cphvb_type.CPHVB_UINT16;
                else if (v is cphvb_uint32)
                    this.type = cphvb_type.CPHVB_UINT32;
                else if (v is cphvb_uint64)
                    this.type = cphvb_type.CPHVB_UINT64;
                else if (v is cphvb_float32)
                    this.type = cphvb_type.CPHVB_FLOAT32;
                else if (v is cphvb_float64)
                    this.type = cphvb_type.CPHVB_FLOAT64;
                else if (v is cphvb_complex64)
                    this.type = cphvb_type.CPHVB_COMPLEX64;
                else if (v is cphvb_complex128)
                    this.type = cphvb_type.CPHVB_COMPLEX128;
                else
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        /// Struct for typesafe assignment of a constant value
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        public struct cphvb_constant_value
        {
            /// <summary>
            /// The boolean value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_bool     bool8;
            /// <summary>
            /// The int8 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_int8     int8;
            /// <summary>
            /// The int16 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_int16    int16;
            /// <summary>
            /// The int32 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_int32    int32;
            /// <summary>
            /// The int64 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_int64    int64;
            /// <summary>
            /// The uint8 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_uint8    uint8;
            /// <summary>
            /// The uin16 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_uint16   uint16;
            /// <summary>
            /// The uint32 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_uint32   uint32;
            /// <summary>
            /// The uint64 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_uint64   uint64;
            /// <summary>
            /// The float32 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_float32  float32;
            /// <summary>
            /// The float64 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_float64  float64;
            /// <summary>
            /// The complex64 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_complex64  complex64;
            /// <summary>
            /// The complex128 value
            /// </summary>
            [FieldOffset(0)] 
            public cphvb_complex128  complex128;

            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_bool v) { this.bool8 = v; return this; }
            //public cphvb_constant Set(cphvb_int8 v) { this.int8 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_int16 v) { this.int16 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_int32 v) { this.int32 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_int64 v) { this.int64 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_uint8 v) { this.uint8 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_uint16 v) { this.uint16 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_uint32 v) { this.uint32 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_uint64 v) { this.uint64 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_float32 v) { this.float32 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_float64 v) { this.float64 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_complex64 v) { this.complex64 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(cphvb_complex128 v) { this.complex128 = v; return this; }
            /// <summary>
            /// Sets the value of this constant
            /// </summary>
            /// <param name="v">The value to set</param>
            /// <returns>A constant struct representing the value</returns>
            public cphvb_constant_value Set(object v) 
            {
                if (v is cphvb_bool)
                    return Set((cphvb_bool)v);
                else if (v is cphvb_int16)
                    return Set((cphvb_int16)v);
                else if (v is cphvb_int32)
                    return Set((cphvb_int32)v);
                else if (v is cphvb_int64)
                    return Set((cphvb_int64)v);
                else if (v is cphvb_uint8)
                    return Set((cphvb_uint8)v);
                else if (v is cphvb_uint16)
                    return Set((cphvb_uint16)v);
                else if (v is cphvb_uint32)
                    return Set((cphvb_uint32)v);
                else if (v is cphvb_uint64)
                    return Set((cphvb_uint64)v);
                else if (v is cphvb_float32)
                    return Set((cphvb_float32)v);
                else if (v is cphvb_float64)
                    return Set((cphvb_float64)v);
                else if (v is cphvb_complex64)
                    return Set((cphvb_complex64)v);
                else if (v is cphvb_complex128)
                    return Set((cphvb_complex128)v);

                throw new NotSupportedException(); 
            }                
        }

        /// <summary>
        /// Represents a native data array
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        public struct cphvb_data_array
        {
//Fix compiler reporting these as unused as they are weirdly mapped,
//and only processed from unmanaged code
#pragma warning disable 0414 
#pragma warning disable 0169
            [FieldOffset(0)] private cphvb_bool[]     bool8;
            [FieldOffset(0)] private cphvb_int8[]     int8;
            [FieldOffset(0)] private cphvb_int16[]    int16;
            [FieldOffset(0)] private cphvb_int32[]    int32;
            [FieldOffset(0)] private cphvb_int64[]    int64;
            [FieldOffset(0)] private cphvb_uint8[]    uint8;
            [FieldOffset(0)] private cphvb_uint16[]   uint16;
            [FieldOffset(0)] private cphvb_uint32[]   uint32;
            [FieldOffset(0)] private cphvb_uint64[]   uint64;
            [FieldOffset(0)] private cphvb_float32[]  float32;
            [FieldOffset(0)] private cphvb_float64[]  float64;
            [FieldOffset(0)] private cphvb_complex64[] complex64;
            [FieldOffset(0)] private cphvb_complex128[] complex128;
            [FieldOffset(0)] private IntPtr voidPtr;
#pragma warning restore 0414
#pragma warning restore 0169

            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_bool[] v) { this.bool8 = v; return this; }
            //public cphvb_data_array Set(cphvb_int8[] v) { this.int8 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_int16[] v) { this.int16 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_int32[] v) { this.int32 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_int64[] v) { this.int64 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_uint8[] v) { this.uint8 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_uint16[] v) { this.uint16 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_uint32[] v) { this.uint32 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_uint64[] v) { this.uint64 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_float32[] v) { this.float32 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_float64[] v) { this.float64 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_complex64[] v) { this.complex64 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(cphvb_complex128[] v) { this.complex128 = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(IntPtr v) { this.voidPtr = v; return this; }
            /// <summary>
            /// Sets the array using a managed array
            /// </summary>
            /// <param name="v">The array to marshal</param>
            /// <returns>This array representation</returns>
            public cphvb_data_array Set(object v) { throw new NotSupportedException(); }
        }

        /// <summary>
        /// A cphVB component
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_component
        {
            /// <summary>
            /// The name of the component
            /// </summary>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst=CPHVB_COMPONENT_NAME_SIZE)]
            public byte[] name;
            /// <summary>
            /// The copmponent configuration dictionary
            /// </summary>
            public IntPtr config;
            /// <summary>
            /// A handle to the dll/so that implements the component
            /// </summary>
            public IntPtr lib_handle;
            /// <summary>
            /// The component type
            /// </summary>
            public cphvb_component_type type;
            /// <summary>
            /// The initialization function
            /// </summary>
            public cphvb_init init;
            /// <summary>
            /// The shutdown function
            /// </summary>
            public cphvb_shutdown shutdown;
            /// <summary>
            /// The execute function
            /// </summary>
            public cphvb_execute execute;
            /// <summary>
            /// The userfunc registration function
            /// </summary>
            public cphvb_reg_func reg_func;
#if DEBUG
            /// <summary>
            /// Converts the Asciiz name to a string, used for debugging only
            /// </summary>
            public string Name { get { return System.Text.Encoding.ASCII.GetString(this.name.TakeWhile(b => !b.Equals(0)).ToArray()); } }
#endif
        }

        /// <summary>
        /// Fake wrapper struct to keep a pointer to cphvb_array typesafe
        /// </summary>
        [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_array_ptr
        {
            /// <summary>
            /// The actual IntPtr value
            /// </summary>
            [FieldOffset(0)]
            internal IntPtr m_ptr;

            /// <summary>
            /// Accessor methods to read/write the data pointer
            /// </summary>
            public IntPtr Data 
            {
                get 
                {
                    if (m_ptr == IntPtr.Zero)
                        throw new ArgumentNullException();

                    //IntPtr test = Marshal.ReadIntPtr(m_ptr, (Marshal.SizeOf(cphvb_intp) * (4 + (CPHVB_MAXDIM * 2))));

                    IntPtr res;
                    cphvb_error e = cphvb_data_get(this, out res);
                    if (e != cphvb_error.CPHVB_SUCCESS)
                        throw new cphVBException(e);
                    return res;
                }
                set
                {
                    if (m_ptr == IntPtr.Zero)
                        throw new ArgumentNullException();

                    cphvb_error e = cphvb_data_set(this, value);
                    if (e != cphvb_error.CPHVB_SUCCESS)
                        throw new cphVBException(e);
                }
            }

            /// <summary>
            /// Accessor methods to read/write the base array
            /// </summary>
            public cphvb_array_ptr BaseArray
            {
                get
                {
                    if (m_ptr == IntPtr.Zero)
                        throw new ArgumentNullException();
                    
                    return new cphvb_array_ptr() { 
                        m_ptr = Marshal.ReadIntPtr(m_ptr, 0)
                    };
                }
            }

            /// <summary>
            /// A value that represents a null pointer
            /// </summary>
            public static readonly cphvb_array_ptr Null = new cphvb_array_ptr() { m_ptr = IntPtr.Zero };

            /// <summary>
            /// Free's the array view, but does not de-reference it with the VEM
            /// </summary>
            public void Free()
            {
                if (m_ptr == IntPtr.Zero)
                    return;

                cphvb_component_free_ptr(m_ptr);
                m_ptr = IntPtr.Zero;
            }

            /// <summary>
            /// Custom equals functionality
            /// </summary>
            /// <param name="obj">The object to compare to</param>
            /// <returns>True if the objects are equal, false otherwise</returns>
            public override bool Equals(object obj)
            {
                if (obj is cphvb_array_ptr)
                    return ((cphvb_array_ptr)obj).m_ptr == this.m_ptr;
                else
                    return base.Equals(obj);
            }

            /// <summary>
            /// Custom hashcode functionality
            /// </summary>
            /// <returns>The hash code for this instance</returns>
            public override cphvb_int32 GetHashCode()
            {
                return m_ptr.GetHashCode();
            }

            /// <summary>
            /// Simple compare operator for pointer type
            /// </summary>
            /// <param name="a">One argument</param>
            /// <param name="b">Another argument</param>
            /// <returns>True if the arguments are the same, false otherwise</returns>
            public static bool operator ==(cphvb_array_ptr a, cphvb_array_ptr b)
            {
                return a.m_ptr == b.m_ptr;
            }

            /// <summary>
            /// Simple compare operator for pointer type
            /// </summary>
            /// <param name="a">One argument</param>
            /// <param name="b">Another argument</param>
            /// <returns>False if the arguments are the same, true otherwise</returns>
            public static bool operator !=(cphvb_array_ptr a, cphvb_array_ptr b)
            {
                return a.m_ptr != b.m_ptr;
            }

            /// <summary>
            /// Returns a human readable string representation of the pointer
            /// </summary>
            /// <returns>A human readable string representation of the pointer</returns>
            public override string ToString()
            {
                return string.Format("(self: {0}, data: {1}, base: {2})", m_ptr, m_ptr == IntPtr.Zero ? "null" : this.Data.ToString(), m_ptr == IntPtr.Zero ? "null" : (this.BaseArray == cphvb_array_ptr.Null ? "null" : this.BaseArray.ToString()));
            }
        }

        /// <summary>
        /// Representation of a cphVB array
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_array
        {
            /// <summary>
            /// The base array if this is a view, null otherwise
            /// </summary>
            public cphvb_array_ptr basearray;
            /// <summary>
            /// The element datatype of the array
            /// </summary>
            public cphvb_type type;
            /// <summary>
            /// The number of dimensions in the array
            /// </summary>
            public cphvb_intp ndim;
            /// <summary>
            /// The data offset
            /// </summary>
            public cphvb_index start;
            /// <summary>
            /// The dimension sizes
            /// </summary>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst=CPHVB_MAXDIM)]
            public cphvb_index[] shape;
            /// <summary>
            /// The dimension strides
            /// </summary>
            [MarshalAs(UnmanagedType.ByValArray, SizeConst=CPHVB_MAXDIM)]
            public cphvb_index[] stride;
            /// <summary>
            /// A pointer to the actual data elements
            /// </summary>
            public cphvb_data_array data;
        }

        /// <summary>
        /// This struct is used to allow us to pass a pointer to different struct types,
        /// because we cannot use inheritance for the cphvb_userfunc structure to
        /// support the reduce structure. Downside is that the size of the struct
        /// will always be the size of the largest one
        /// </summary>
        [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_union
        {
            /// <summary>
            /// The plain userfunc
            /// </summary>
            [FieldOffset(0)]
            public cphvb_userfunc_plain plain;

            /// <summary>
            /// The random userfunc
            /// </summary>
            [FieldOffset(0)]
            public cphvb_userfunc_random random;

            /// <summary>
            /// The reduce userfunc
            /// </summary>
            [FieldOffset(0)]
            public cphvb_userfunc_reduce reduce;

            /// <summary>
            /// The matmul userfunc
            /// </summary>
            [FieldOffset(0)]
            public cphvb_userfunc_matmul matmul;

            /// <summary>
            /// Constructs a new union representin a plain userfunc
            /// </summary>
            /// <param name="arg">The user defined function</param>
            public cphvb_userfunc_union(cphvb_userfunc_plain arg) : this() { plain = arg; }
            /// <summary>
            /// Constructs a new union representin a reduce userfunc
            /// </summary>
            /// <param name="arg">The user defined function</param>
            public cphvb_userfunc_union(cphvb_userfunc_reduce arg) : this() { reduce = arg; }
            /// <summary>
            /// Constructs a new union representin a random userfunc
            /// </summary>
            /// <param name="arg">The user defined function</param>
            public cphvb_userfunc_union(cphvb_userfunc_random arg) : this() { random = arg; }
            /// <summary>
            /// Constructs a new union representin a matmul userfunc
            /// </summary>
            /// <param name="arg">The user defined function</param>
            public cphvb_userfunc_union(cphvb_userfunc_matmul arg) : this() { matmul = arg; }

            /// <summary>
            /// Implicit operator for creating a union with a plain userfunc
            /// </summary>
            /// <param name="arg">The userfunc</param>
            /// <returns>The union userfunc</returns>
            public static implicit operator cphvb_userfunc_union(cphvb_userfunc_plain arg) { return new cphvb_userfunc_union(arg); }
            /// <summary>
            /// Implicit operator for creating a union with a reduce userfunc
            /// </summary>
            /// <param name="arg">The userfunc</param>
            /// <returns>The union userfunc</returns>
            public static implicit operator cphvb_userfunc_union(cphvb_userfunc_reduce arg) { return new cphvb_userfunc_union(arg); }
            /// <summary>
            /// Implicit operator for creating a union with a random userfunc
            /// </summary>
            /// <param name="arg">The userfunc</param>
            /// <returns>The union userfunc</returns>
            public static implicit operator cphvb_userfunc_union(cphvb_userfunc_random arg) { return new cphvb_userfunc_union(arg); }
            /// <summary>
            /// Implicit operator for creating a union with a matmul userfunc
            /// </summary>
            /// <param name="arg">The userfunc</param>
            /// <returns>The union userfunc</returns>
            public static implicit operator cphvb_userfunc_union(cphvb_userfunc_matmul arg) { return new cphvb_userfunc_union(arg); }
        }

        /// <summary>
        /// The reduce userdefined function
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_reduce
        {
            /// <summary>
            /// The reduce function id
            /// </summary>
            public cphvb_intp id;
            /// <summary>
            /// The number of output elements
            /// </summary>
            public cphvb_intp nout;
            /// <summary>
            /// The number of input elements
            /// </summary>
            public cphvb_intp nin;
            /// <summary>
            /// The total size of this struct
            /// </summary>
            public cphvb_intp struct_size;
            /// <summary>
            /// The output operand
            /// </summary>
            public cphvb_array_ptr operand0;
            /// <summary>
            /// The input operand
            /// </summary>
            public cphvb_array_ptr operand1;
            /// <summary>
            /// The axis to reduce over
            /// </summary>
            public cphvb_index axis;
            /// <summary>
            /// The opcode for the binary function used to reduce
            /// </summary>
            public cphvb_opcode opcode;

            /// <summary>
            /// Constructs a new reduce userfunc
            /// </summary>
            /// <param name="func">The id for the reduce userfunc</param>
            /// <param name="opcode">The opcode for the binary function used to reduce with</param>
            /// <param name="axis">The axis to reduce</param>
            /// <param name="op1">The output operand</param>
            /// <param name="op2">The input operand</param>
            public cphvb_userfunc_reduce(cphvb_intp func, cphvb_opcode opcode, cphvb_intp axis, cphvb_array_ptr op1, cphvb_array_ptr op2)
            {
                this.id = func;
                this.nout = 1;
                this.nin = 1;
                this.struct_size = REDUCEFUNC_SIZE;
                this.operand0 = op1;
                this.operand1 = op2;
                this.axis = axis;
                this.opcode = opcode;
            }
        }

        /// <summary>
        /// The random userfunc
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_random
        {
            /// <summary>
            /// The random function id
            /// </summary>
            public cphvb_intp id;
            /// <summary>
            /// The number of output elements
            /// </summary>
            public cphvb_intp nout;
            /// <summary>
            /// The number of input elements
            /// </summary>
            public cphvb_intp nin;
            /// <summary>
            /// The total size of this struct
            /// </summary>
            public cphvb_intp struct_size;
            /// <summary>
            /// The output operand
            /// </summary>
            public cphvb_array_ptr operand;

            /// <summary>
            /// Creates a new random userfunc
            /// </summary>
            /// <param name="func">The random function id</param>
            /// <param name="op">The output operand</param>
            public cphvb_userfunc_random(cphvb_intp func, cphvb_array_ptr op)
            {
                this.id = func;
                this.nout = 1;
                this.nin = 0;
                this.struct_size = RANDOMFUNC_SIZE;
                this.operand = op;
            }
        }

        /// <summary>
        /// The matmul userfunc
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_matmul
        {
            /// <summary>
            /// The matmul function id
            /// </summary>
            public cphvb_intp id;
            /// <summary>
            /// The number of output operands
            /// </summary>
            public cphvb_intp nout;
            /// <summary>
            /// The number of input operands
            /// </summary>
            public cphvb_intp nin;
            /// <summary>
            /// The total size of this struct
            /// </summary>
            public cphvb_intp struct_size;
            /// <summary>
            /// The output operand
            /// </summary>
            public cphvb_array_ptr operand0;
            /// <summary>
            /// An input operand
            /// </summary>
            public cphvb_array_ptr operand1;
            /// <summary>
            /// Another input operand
            /// </summary>
            public cphvb_array_ptr operand2;

            /// <summary>
            /// Constructs a new matmul userfunc
            /// </summary>
            /// <param name="func">The matmul function id</param>
            /// <param name="op1">The output operand</param>
            /// <param name="op2">An input operand</param>
            /// <param name="op3">Another input operand</param>
            public cphvb_userfunc_matmul(cphvb_intp func, cphvb_array_ptr op1, cphvb_array_ptr op2, cphvb_array_ptr op3)
            {
                this.id = func;
                this.nout = 1;
                this.nin = 2;
                this.struct_size = MATMULFUNC_SIZE;
                this.operand0 = op1;
                this.operand1 = op2;
                this.operand2 = op3;
            }
        }

        /// <summary>
        /// A plain userfunc
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_plain
        {
            /// <summary>
            /// The function id
            /// </summary>
            public cphvb_intp id;
            /// <summary>
            /// The number of output operands
            /// </summary>
            public cphvb_intp nout;
            /// <summary>
            /// The number of input operands
            /// </summary>
            public cphvb_intp nin;
            /// <summary>
            /// The total size of the struct
            /// </summary>
            public cphvb_intp struct_size;
            /// <summary>
            /// The output operand
            /// </summary>
            public cphvb_array_ptr operand0;
            /// <summary>
            /// An input operand
            /// </summary>
            public cphvb_array_ptr operand1;
            /// <summary>
            /// Another input operand
            /// </summary>
            public cphvb_array_ptr operand2;

            /// <summary>
            /// Creates a new plain userfunc
            /// </summary>
            /// <param name="func">The function id</param>
            /// <param name="op">The output operand</param>
            public cphvb_userfunc_plain(cphvb_intp func, cphvb_array_ptr op)
            {
                this.id = func;
                this.nout = 1;
                this.nin = 0;
                this.struct_size = PLAINFUNC_SIZE;
                this.operand0 = op;
                this.operand1 = cphvb_array_ptr.Null;
                this.operand2 = cphvb_array_ptr.Null;
            }

            /// <summary>
            /// Creates a new plain userfunc
            /// </summary>
            /// <param name="func">The function id</param>
            /// <param name="op1">The output operand</param>
            /// <param name="op2">The input operand</param>
            public cphvb_userfunc_plain(cphvb_intp func, cphvb_array_ptr op1, cphvb_array_ptr op2)
            {
                this.id = func;
                this.nout = 1;
                this.nin = 0;
                this.struct_size = PLAINFUNC_SIZE;
                this.operand0 = op1;
                this.operand1 = op2;
                this.operand2 = cphvb_array_ptr.Null;
            }

            /// <summary>
            /// Creates a new plain userfunc
            /// </summary>
            /// <param name="func">The function id</param>
            /// <param name="op1">The output operand</param>
            /// <param name="op2">An input operand</param>
            /// <param name="op3">Another input operand</param>
            public cphvb_userfunc_plain(cphvb_intp func, cphvb_array_ptr op1, cphvb_array_ptr op2, cphvb_array_ptr op3)
            {
                this.id = func;
                this.nout = 1;
                this.nin = 0;
                this.struct_size = PLAINFUNC_SIZE;
                this.operand0 = op1;
                this.operand1 = op2;
                this.operand2 = op3;
            }
        }

        /// <summary>
        /// Represents a cphVB instruction
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_instruction : IInstruction
        {
            /// <summary>
            /// Instruction status
            /// </summary>
            public cphvb_error status;
            /// <summary>
            /// The instruction opcode
            /// </summary>
            public cphvb_opcode opcode;
            /// <summary>
            /// The output operand
            /// </summary>
            public cphvb_array_ptr operand0;
            /// <summary>
            /// An input operand
            /// </summary>
            public cphvb_array_ptr operand1;
            /// <summary>
            /// Another input operand
            /// </summary>
            public cphvb_array_ptr operand2;
            /// <summary>
            /// A constant value assigned to the instruction
            /// </summary>
            public cphvb_constant constant;     
            /// <summary>
            /// Points to the user-defined function when the opcode is CPHVB_USERFUNC
            /// </summary>
            public IntPtr userfunc;

            /// <summary>
            /// Creates a new instruction
            /// </summary>
            /// <param name="opcode">The opcode for the operation</param>
            /// <param name="operand">The output operand</param>
            /// <param name="constant">An optional constant</param>
            public cphvb_instruction(cphvb_opcode opcode, cphvb_array_ptr operand, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
            {
                this.status = cphvb_error.CPHVB_INST_PENDING;
                this.opcode = opcode;
                this.operand0 = operand;
                this.operand1 = cphvb_array_ptr.Null;
                this.operand2 = cphvb_array_ptr.Null;
                this.userfunc = IntPtr.Zero;
                this.constant = constant;
            }

            /// <summary>
            /// Creates a new instruction
            /// </summary>
            /// <param name="opcode">The opcode for the operation</param>
            /// <param name="operand1">The output operand</param>
            /// <param name="constant">A left-hand-side constant</param>
            /// <param name="operand2">An input operand</param>
            public cphvb_instruction(cphvb_opcode opcode, cphvb_array_ptr operand1, PInvoke.cphvb_constant constant, cphvb_array_ptr operand2)
            {
                this.status = cphvb_error.CPHVB_INST_PENDING;
                this.opcode = opcode;
                this.operand0 = operand1;
                this.operand1 = cphvb_array_ptr.Null;
                this.operand2 = operand2;
                this.userfunc = IntPtr.Zero;
                this.constant = constant;
            }

            /// <summary>
            /// Creates a new instruction
            /// </summary>
            /// <param name="opcode">The opcode for the operation</param>
            /// <param name="operand1">The output operand</param>
            /// <param name="operand2">An input operand</param>
            /// <param name="constant">A right-hand-side constant</param>
            public cphvb_instruction(cphvb_opcode opcode, cphvb_array_ptr operand1, cphvb_array_ptr operand2, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
            {
                this.status = cphvb_error.CPHVB_INST_PENDING;
                this.opcode = opcode;
                this.operand0 = operand1;
                this.operand1 = operand2;
                this.operand2 = cphvb_array_ptr.Null;
                this.userfunc = IntPtr.Zero;
                this.constant = constant;
            }

            /// <summary>
            /// Creates a new instruction
            /// </summary>
            /// <param name="opcode">The opcode for the operation</param>
            /// <param name="operand1">The output operand</param>
            /// <param name="operand2">An input operand</param>
            /// <param name="operand3">Another input operand</param>
            /// <param name="constant">A right-hand-side constant</param>
            public cphvb_instruction(cphvb_opcode opcode, cphvb_array_ptr operand1, cphvb_array_ptr operand2, cphvb_array_ptr operand3, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
            {
                this.status = cphvb_error.CPHVB_INST_PENDING;
                this.opcode = opcode;
                this.operand0 = operand1;
                this.operand1 = operand2;
                this.operand2 = operand3;
                this.userfunc = IntPtr.Zero;
                this.constant = constant;
            }

            /// <summary>
            /// Creates a new instruction
            /// </summary>
            /// <param name="opcode">The opcode for the operation</param>
            /// <param name="operands">A list of operands</param>
            /// <param name="constant">A constant</param>
            public cphvb_instruction(cphvb_opcode opcode, IEnumerable<cphvb_array_ptr> operands, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
            {
                this.status = cphvb_error.CPHVB_INST_PENDING;
                this.opcode = opcode;
                var en = operands.GetEnumerator();
                if (en.MoveNext())
                {
                    this.operand0 = en.Current;
                    if (en.MoveNext())
                    {
                        this.operand1 = en.Current;
                        if (en.MoveNext())
                            this.operand2 = en.Current;
                        else
                            this.operand2 = cphvb_array_ptr.Null;
                    }
                    else
                    {
                        this.operand1 = cphvb_array_ptr.Null;
                        this.operand2 = cphvb_array_ptr.Null;
                    }
                }
                else
                {
                    this.operand0 = cphvb_array_ptr.Null;
                    this.operand1 = cphvb_array_ptr.Null;
                    this.operand2 = cphvb_array_ptr.Null;
                }
                this.userfunc = IntPtr.Zero;
                this.constant = constant;
            }

            /// <summary>
            /// Constructs a userdefined instruction
            /// </summary>
            /// <param name="opcode">The opcode CPHVB_USERFUNC</param>
            /// <param name="userfunc">A pointer to the userfunc struct</param>
            public cphvb_instruction(cphvb_opcode opcode, IntPtr userfunc)
            {
                this.status = cphvb_error.CPHVB_INST_PENDING;
                this.opcode = opcode;
                this.userfunc = userfunc;
                this.operand0 = cphvb_array_ptr.Null;
                this.operand1 = cphvb_array_ptr.Null;
                this.operand2 = cphvb_array_ptr.Null;
                this.constant = new cphvb_constant();
            }

            /// <summary>
            /// Returns a human readable representation of the instruction
            /// </summary>
            /// <returns>A human readable representation of the instruction</returns>
            public override string ToString()
            {
                return string.Format("{0}({1}, {2}, {3})", this.opcode, operand0, operand1, operand2);
            }

            cphvb_opcode IInstruction.OpCode
            {
                get { return opcode; }
            }
        }

        /// <summary>
        /// Delegate for initializing a cphVB component
        /// </summary>
        /// <param name="self">An allocated component struct that gets filled with data</param>
        /// <returns>A status code</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_init(ref cphvb_component self);
        /// <summary>
        /// Delegate for shutting down a component
        /// </summary>
        /// <returns>A status code</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_shutdown();
        /// <summary>
        /// Delegate for execution instructions
        /// </summary>
        /// <param name="count">The number of instructions to execute</param>
        /// <param name="inst_list">The list of instructions to execute</param>
        /// <returns>A status code</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_execute(cphvb_intp count, [In, Out]cphvb_instruction[] inst_list);
        /// <summary>
        /// Register a userfunc
        /// </summary>
        /// <param name="fun">The name of the function to register</param>
        /// <param name="id">The id assigned</param>
        /// <returns>A status code</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_reg_func(string fun, ref cphvb_intp id);
        
        /// <summary>
        /// Creates a new base array or view in cphVB
        /// </summary>
        /// <param name="basearray">The base array if creating a view, null otherwise</param>
        /// <param name="type">The element datatype for the array</param>
        /// <param name="ndim">The number of dimensions</param>
        /// <param name="start">The data pointer offset</param>
        /// <param name="shape">The size of each dimension</param>
        /// <param name="stride">The stride of each dimension</param>
        /// <param name="new_array">The allocated array</param>
        /// <returns>A status code</returns>
        [DllImport("libcphvb", EntryPoint = "cphvb_create_array", CallingConvention = CallingConvention.Cdecl, SetLastError = true, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_create_array(
                                   cphvb_array_ptr basearray,
                                   cphvb_type     type,
                                   cphvb_intp     ndim,
                                   cphvb_index    start,
                                   cphvb_index[]    shape,
                                   cphvb_index[]    stride,
                                   out cphvb_array_ptr new_array);

        /// <summary>
        /// Deallocates metadata for a base array or view
        /// </summary>
        /// <param name="array">The array to deallocate</param>
        /// <returns>A status code</returns>
        [DllImport("libcphvb", EntryPoint = "cphvb_destroy_array", CallingConvention = CallingConvention.Cdecl, SetLastError = true, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_destroy_array(cphvb_array_ptr array);

        /// <summary>
        /// Setup the root component, which normally is the bridge.
        /// </summary>
        /// <returns>A new component object</returns>
        [DllImport("libcphvb", EntryPoint = "cphvb_component_setup", CallingConvention = CallingConvention.Cdecl, SetLastError = true, CharSet = CharSet.Auto)]
        private extern static IntPtr cphvb_component_setup_masked();

        /// <summary>
        /// Setup the root component, which normally is the bridge.
        /// </summary>
        /// <returns>A new component object</returns>
        public static cphvb_component cphvb_component_setup(out IntPtr unmanaged)
        {
            unmanaged = cphvb_component_setup_masked();
            cphvb_component r = (cphvb_component)Marshal.PtrToStructure(unmanaged, typeof(cphvb_component));
			return r;
        }

        /// <summary>
        /// Retrieves the children components of the parent.
        /// NB: the array and all the children should be free'd by the caller
        /// </summary>
        /// <param name="parent">The parent component (input)</param>
        /// <param name="count">Number of children components</param>
        /// <param name="children">Array of children components (output)</param>
        /// <returns>Error code (CPHVB_SUCCESS)</returns>
        [DllImport("libcphvb", EntryPoint = "cphvb_component_children", CallingConvention = CallingConvention.Cdecl, SetLastError = true, CharSet = CharSet.Auto)]
        private extern static cphvb_error cphvb_component_children_masked([In] ref cphvb_component parent, [Out] out cphvb_intp count, [Out] out IntPtr children);

        /// <summary>
        /// Retrieves the children components of the parent.
        /// NB: the array and all the children should be free'd by the caller
        /// </summary>
        /// <param name="parent">The parent component (input)</param>
        /// <param name="unmanagedData">Unmanaged data</param>
        /// <param name="children">Array of children components (output)</param>
        /// <returns>Error code (CPHVB_SUCCESS)</returns>
        public static cphvb_error cphvb_component_children(cphvb_component parent, out cphvb_component[] children, out IntPtr unmanagedData)
        {
            //TODO: Errors in setup may cause memory leaks, but we should terminate anyway

            long count = 0;
            children = null;

            cphvb_error e = cphvb_component_children_masked(ref parent, out count, out unmanagedData);
            if (e != cphvb_error.CPHVB_SUCCESS)
                return e;

            children = new cphvb_component[count];
            for (int i = 0; i < count; i++)
            {
                IntPtr cur = Marshal.ReadIntPtr(unmanagedData, Marshal.SizeOf(typeof(cphvb_intp)) * i);
                children[i] = (cphvb_component)Marshal.PtrToStructure(cur, typeof(cphvb_component));
            }

            return e;
        }


        /// <summary>
        /// Frees the component
        /// </summary>
        /// <param name="component">The component to free</param>
        /// <returns>Error code (CPHVB_SUCCESS)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_component_free([In] ref cphvb_component component);

        /// <summary>
        /// Frees the component
        /// </summary>
        /// <param name="component">The component to free</param>
        /// <returns>Error code (CPHVB_SUCCESS)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_component_free(IntPtr component);

        /// <summary>
        /// Frees the component
        /// </summary>
        /// <param name="component">The component to free</param>
        /// <returns>Error code (CPHVB_SUCCESS)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_component_free_ptr([In] IntPtr component);
        
        /// <summary>
        /// Retrieves an user-defined function
        /// </summary>
        /// <param name="self">The component</param>
        /// <param name="func">Name of the function e.g. myfunc</param>
        /// <param name="ret_func">Pointer to the function (output), Is NULL if the function doesn't exist</param>
        /// <returns>Error codes (CPHVB_SUCCESS)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_component_get_func([In] ref cphvb_component self, [In] string func,
                               [Out] IntPtr ret_func);

        /// <summary>
        /// Trace an array creation
        /// </summary>
        /// <param name="self">The component</param>
        /// <param name="ary">The array to trace</param>
        /// <returns>Error code (CPHVB_SUCCESS)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_component_trace_array([In] ref cphvb_component self, [In] ref cphvb_array ary);


        /// <summary>
        /// Trace an instruction
        /// </summary>
        /// <param name="self">The component</param>
        /// <param name="inst">The instruction to trace</param>
        /// <returns>Error code (CPHVB_SUCCESS)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_component_trace_inst([In] ref cphvb_component self, [In] ref cphvb_instruction inst);

        /// <summary>
        /// Set the data pointer for the array.
        /// Can only set to non-NULL if the data ptr is already NULL
        /// </summary>
        /// <param name="array">The array in question</param>
        /// <param name="data">The new data pointer</param>
        /// <returns>Error code (CPHVB_SUCCESS, CPHVB_ERROR)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_data_set([In] cphvb_array_ptr array, [In] IntPtr data);

        /// <summary>
        /// Set the data pointer for the array.
        /// Can only set to non-NULL if the data ptr is already NULL
        /// </summary>
        /// <param name="array">The array in question</param>
        /// <returns>Error code (CPHVB_SUCCESS, CPHVB_ERROR)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_data_malloc([In] cphvb_array_ptr array);

        /// <summary>
        /// Set the data pointer for the array.
        /// Can only set to non-NULL if the data ptr is already NULL
        /// </summary>
        /// <param name="array">The array in question</param>
        /// <returns>Error code (CPHVB_SUCCESS, CPHVB_ERROR)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_data_free([In] cphvb_array_ptr array);

        /// <summary>
        /// Get the data pointer for the array.
        /// </summary>
        /// <param name="array">The array in question</param>
        /// <param name="data">The data pointer</param>
        /// <returns>Error code (CPHVB_SUCCESS, CPHVB_ERROR)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_data_get([In] cphvb_array_ptr array, [Out] out IntPtr data);

    }
}
