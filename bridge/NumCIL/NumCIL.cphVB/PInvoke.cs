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
using cphvb_float32 = System.Single;
using cphvb_float64 = System.Double;

namespace NumCIL.cphVB
{
    /*internal class InstructionMarshal : ICustomMarshaler
    {
        private static readonly int SIZE = Marshal.SizeOf(typeof(PInvoke.cphvb_instruction));
        private static readonly InstructionMarshal Instance = new InstructionMarshal();

        public static ICustomMarshaler GetInstance(string arg)
        {
            return Instance;
        }

        public void CleanUpManagedData(object ManagedObj)
        {
        }

        public void CleanUpNativeData(IntPtr pNativeData)
        {
            if (pNativeData == IntPtr.Zero)
                return;
            Marshal.FreeHGlobal(pNativeData);
            
        }

        public int GetNativeDataSize()
        {
            return SIZE;
        }

        public IntPtr MarshalManagedToNative(object ManagedObj)
        {
            System.Diagnostics.Debug.Assert(PInvoke.INTP_SIZE == 8);
            System.Diagnostics.Debug.Assert(SIZE == 48);

            PInvoke.cphvb_instruction[] arg = (PInvoke.cphvb_instruction[])ManagedObj;
            int userfuncs = 0;
            int instructions = 0;

            foreach (var a in arg)
                if (a.operand0 == PInvoke.cphvb_array_ptr.Null)
                    break;
                else
                {
                    instructions++;
                    if (a.userfunc != null)
                        userfuncs++;
                }

            //Allocate a single large chunck with all the data, including custom function descriptors
            IntPtr chunck = Marshal.AllocHGlobal((instructions * SIZE) + (userfuncs * PInvoke.USERFUNC_SIZE));
            IntPtr cur = chunck;
            IntPtr curUser = IntPtr.Add(cur, (instructions * SIZE));

            for(int i = 0; i < instructions; i++)
            {
                PInvoke.cphvb_instruction mo = arg[i];

                Marshal.WriteInt64(cur, 0, (long)mo.status);
                Marshal.WriteInt64(cur, 8, (long)mo.opcode);
                Marshal.WriteIntPtr(cur, 16, mo.operand0.m_ptr);
                Marshal.WriteIntPtr(cur, 24, mo.operand1.m_ptr);
                Marshal.WriteIntPtr(cur, 32, mo.operand2.m_ptr);

                if (mo.userfunc == null)
                {
                    Marshal.WriteIntPtr(cur, 40, IntPtr.Zero);
                }
                else
                {
                    Marshal.StructureToPtr(mo.userfunc, curUser, false);
                    Marshal.WriteIntPtr(cur, 40, curUser);
                    curUser = IntPtr.Add(curUser, PInvoke.USERFUNC_SIZE);
                }

                cur = IntPtr.Add(cur, SIZE);
            }

            return chunck;
        }

        public object MarshalNativeToManaged(IntPtr pNativeData)
        {
            throw new NotImplementedException();
        }
    }*/

    public static class PInvoke
    {
        public const int CPHVB_COMPONENT_NAME_SIZE = 1024;
        public const int CPHVB_MAXDIM = 16;
        public const int CPHVB_MAX_EXTRA_META_DATA = 1024;
        public const int CPHVB_MAX_NO_OPERANDS = 3;

        public static readonly bool Is64Bit = IntPtr.Size == 8;
        public static readonly int INTP_SIZE = Marshal.SizeOf(typeof(cphvb_intp));
        public static readonly int USERFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_union));
        public static readonly int RANDOMFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_random));
        public static readonly int REDUCEFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_reduce));
        public static readonly int MATMULFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_matmul));
        public static readonly int PLAINFUNC_SIZE = Marshal.SizeOf(typeof(cphvb_userfunc_plain));

        public enum cphvb_component_type : long
        {
            CPHVB_BRIDGE,
            CPHVB_VEM,
            CPHVB_VE,
            CPHVB_COMPONENT_ERROR
        }

        public enum cphvb_error : long
        {
            CPHVB_SUCCESS,
            CPHVB_ERROR,
            CPHVB_TYPE_ERROR,
            CPHVB_TYPE_NOT_SUPPORTED,
            CPHVB_TYPE_NOT_SUPPORTED_BY_OP,
            CPHVB_TYPE_COMB_NOT_SUPPORTED,
            CPHVB_OUT_OF_MEMORY,
            CPHVB_RESULT_IS_CONSTANT,
            CPHVB_OPERAND_UNKNOWN,
            CPHVB_ALREADY_INITALIZED,
            CPHVB_NOT_INITALIZED,
            CPHVB_PARTIAL_SUCCESS,
            CPHVB_INST_DONE,
            CPHVB_INST_UNDONE,
            CPHVB_INST_NOT_SUPPORTED,
            CPHVB_INST_NOT_SUPPORTED_FOR_SLICE,
            CPHVB_USERFUNC_NOT_SUPPORTED,
        }

        public enum cphvb_opcode : long
        {
            CPHVB_ADD,
            CPHVB_SUBTRACT,
            CPHVB_MULTIPLY,
            CPHVB_DIVIDE,
            CPHVB_LOGADDEXP,
            CPHVB_LOGADDEXP2,
            CPHVB_TRUE_DIVIDE,
            CPHVB_FLOOR_DIVIDE,
            CPHVB_NEGATIVE,
            CPHVB_POWER,
            CPHVB_REMAINDER,
            CPHVB_MOD,
            CPHVB_FMOD,
            CPHVB_ABSOLUTE,
            CPHVB_RINT,
            CPHVB_SIGN,
            CPHVB_CONJ,
            CPHVB_EXP,
            CPHVB_EXP2,
            CPHVB_LOG,
            CPHVB_LOG10,
            CPHVB_EXPM1,
            CPHVB_LOG1P,
            CPHVB_SQRT,
            CPHVB_SQUARE,
            CPHVB_RECIPROCAL,
            CPHVB_ONES_LIKE,
            CPHVB_SIN,
            CPHVB_COS,
            CPHVB_TAN,
            CPHVB_ARCSIN,
            CPHVB_ARCCOS,
            CPHVB_ARCTAN,
            CPHVB_ARCTAN2,
            CPHVB_HYPOT,
            CPHVB_SINH,
            CPHVB_COSH,
            CPHVB_TANH,
            CPHVB_ARCSINH,
            CPHVB_ARCCOSH,
            CPHVB_ARCTANH,
            CPHVB_DEG2RAD,
            CPHVB_RAD2DEG,
            CPHVB_BITWISE_AND,
            CPHVB_BITWISE_OR,
            CPHVB_BITWISE_XOR,
            CPHVB_LOGICAL_NOT,
            CPHVB_LOGICAL_AND,
            CPHVB_LOGICAL_OR,
            CPHVB_LOGICAL_XOR,
            CPHVB_INVERT,
            CPHVB_LEFT_SHIFT,
            CPHVB_RIGHT_SHIFT,
            CPHVB_GREATER,
            CPHVB_GREATER_EQUAL,
            CPHVB_LESS,
            CPHVB_LESS_EQUAL,
            CPHVB_NOT_EQUAL,
            CPHVB_EQUAL,
            CPHVB_MAXIMUM,
            CPHVB_MINIMUM,
            CPHVB_ISFINITE,
            CPHVB_ISINF,
            CPHVB_ISNAN,
            CPHVB_SIGNBIT,
            CPHVB_MODF,
            CPHVB_LDEXP,
            CPHVB_FREXP,
            CPHVB_FLOOR,
            CPHVB_CEIL,
            CPHVB_TRUNC,
            CPHVB_LOG2,
            CPHVB_ISREAL,
            CPHVB_ISCOMPLEX,
            CPHVB_IDENTITY,
            CPHVB_USERFUNC,//It is an user-defined function
            CPHVB_SYNC,    //Inform child to make data synchronized and available.
            CPHVB_DISCARD, //Inform child to forget the array
            CPHVB_DESTROY, //Inform child to deallocate the array.
            CPHVB_RANDOM,  //file out with random
            CPHVB_ARANGE, // out, start, step
            //Used by a brigde to mark untranslatable operations.
            //NB: CPHVB_NONE must be the last element in this enum.
            CPHVB_NONE
        }

        public enum cphvb_type : long
        {
            CPHVB_BOOL,
            CPHVB_INT8,
            CPHVB_INT16,
            CPHVB_INT32,
            CPHVB_INT64,
            CPHVB_UINT8,
            CPHVB_UINT16,
            CPHVB_UINT32,
            CPHVB_UINT64,
            CPHVB_FLOAT16,
            CPHVB_FLOAT32,
            CPHVB_FLOAT64,
            CPHVB_INDEX, // Not a data type same as INT32 used for e.g. reduce dim
            //NB: CPHVB_UNKNOWN must be the last element in this enum.
            CPHVB_UNKNOWN
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        struct dictionary
        {
            public int        n ;     /** Number of entries in dictionary */
            public int        size ;  /** Storage size */
            public byte[][]  val ;   /** List of string values */
            public byte[][]    key ;   /** List of string keys */
            public uint[]   hash ;  /** List of hash values for keys */
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct cphvb_constant
        {
            cphvb_constant_value value;
            cphvb_type type;

            public cphvb_constant(cphvb_type type, object v)
            {
                this.type = type;
                this.value = new cphvb_constant_value().Set(v);
            }

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
                else
                    throw new NotSupportedException();
            }
        }

        [StructLayout(LayoutKind.Explicit)]
        public struct cphvb_constant_value
        {
            [FieldOffset(0)] public cphvb_bool     bool8;
            [FieldOffset(0)] public cphvb_int8     int8;
            [FieldOffset(0)] public cphvb_int16    int16;
            [FieldOffset(0)] public cphvb_int32    int32;
            [FieldOffset(0)] public cphvb_int64    int64;
            [FieldOffset(0)] public cphvb_uint8    uint8;
            [FieldOffset(0)] public cphvb_uint16   uint16;
            [FieldOffset(0)] public cphvb_uint32   uint32;
            [FieldOffset(0)] public cphvb_uint64   uint64;
            [FieldOffset(0)] public cphvb_float32  float32;
            [FieldOffset(0)] public cphvb_float64  float64;

            public cphvb_constant_value Set(cphvb_bool v) { this.bool8 = v; return this; }
            //public cphvb_constant Set(cphvb_int8 v) { this.int8 = v; return this; }
            public cphvb_constant_value Set(cphvb_int16 v) { this.int16 = v; return this; }
            public cphvb_constant_value Set(cphvb_int32 v) { this.int32 = v; return this; }
            public cphvb_constant_value Set(cphvb_int64 v) { this.int64 = v; return this; }
            public cphvb_constant_value Set(cphvb_uint8 v) { this.uint8 = v; return this; }
            public cphvb_constant_value Set(cphvb_uint16 v) { this.uint16 = v; return this; }
            public cphvb_constant_value Set(cphvb_uint32 v) { this.uint32 = v; return this; }
            public cphvb_constant_value Set(cphvb_uint64 v) { this.uint64 = v; return this; }
            public cphvb_constant_value Set(cphvb_float32 v) { this.float32 = v; return this; }
            public cphvb_constant_value Set(cphvb_float64 v) { this.float64 = v; return this; }
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

                throw new NotSupportedException(); 
            }                
        }

        [StructLayout(LayoutKind.Explicit)]
        public struct cphvb_data_array
        {
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
            [FieldOffset(0)] private IntPtr           voidPtr;

            public cphvb_data_array Set(cphvb_bool[] v) { this.bool8 = v; return this; }
            //public cphvb_data_array Set(cphvb_int8[] v) { this.int8 = v; return this; }
            public cphvb_data_array Set(cphvb_int16[] v) { this.int16 = v; return this; }
            public cphvb_data_array Set(cphvb_int32[] v) { this.int32 = v; return this; }
            public cphvb_data_array Set(cphvb_int64[] v) { this.int64 = v; return this; }
            public cphvb_data_array Set(cphvb_uint8[] v) { this.uint8 = v; return this; }
            public cphvb_data_array Set(cphvb_uint16[] v) { this.uint16 = v; return this; }
            public cphvb_data_array Set(cphvb_uint32[] v) { this.uint32 = v; return this; }
            public cphvb_data_array Set(cphvb_uint64[] v) { this.uint64 = v; return this; }
            public cphvb_data_array Set(cphvb_float32[] v) { this.float32 = v; return this; }
            public cphvb_data_array Set(cphvb_float64[] v) { this.float64 = v; return this; }
            public cphvb_data_array Set(IntPtr v) { this.voidPtr = v; return this; }
            public cphvb_data_array Set(object v) { throw new NotSupportedException(); }
        }

        /*[StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi)]
        public struct cphvb_execute_union
        {
            [FieldOffset(0)]
            public cphvb_execute execute_normal;
            [FieldOffset(0)]
            public cphvb_execute_with_userfunc execute_with_userfunc;
        }*/

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_component
        {
            [MarshalAs(UnmanagedType.ByValArray, SizeConst=CPHVB_COMPONENT_NAME_SIZE)]
            public byte[] name;
            public IntPtr config;  /*dictionary *config;*/
            public IntPtr lib_handle; //Handle for the dynamic linked library.
            public cphvb_component_type type;
            public cphvb_init init;
            public cphvb_shutdown shutdown;
            public cphvb_execute execute;
            public cphvb_reg_func reg_func;
            public cphvb_create_array create_array; //Only for VEMs

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
                        m_ptr = Marshal.ReadIntPtr(m_ptr, INTP_SIZE)
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

            public override string ToString()
            {
                return string.Format("(self: {0}, data: {1}, base: {2})", m_ptr, m_ptr == IntPtr.Zero ? "null" : this.Data.ToString(), m_ptr == IntPtr.Zero ? "null" : (this.BaseArray == cphvb_array_ptr.Null ? "null" : this.BaseArray.ToString()));
            }
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_array
        {
            public cphvb_intp owner;
            public cphvb_array_ptr basearray;
            public cphvb_type type;
            public cphvb_intp ndim;
            public cphvb_index start;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst=CPHVB_MAXDIM)]
            public cphvb_index[] shape;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst=CPHVB_MAXDIM)]
            public cphvb_index[] stride;
            public cphvb_data_array data;
            public cphvb_intp ref_count;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst=CPHVB_MAX_EXTRA_META_DATA)]
            public byte[] extra_meta_data;
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
            [FieldOffset(0)]
            public cphvb_userfunc_plain plain;

            [FieldOffset(0)]
            public cphvb_userfunc_random random;

            [FieldOffset(0)]
            public cphvb_userfunc_reduce reduce;

            public cphvb_userfunc_union(cphvb_userfunc_plain arg) : this() { plain = arg; }
            public cphvb_userfunc_union(cphvb_userfunc_reduce arg) : this() { reduce = arg; }
            public cphvb_userfunc_union(cphvb_userfunc_random arg) : this() { random = arg; }

            public static implicit operator cphvb_userfunc_union(cphvb_userfunc_plain arg) { return new cphvb_userfunc_union(arg); }
            public static implicit operator cphvb_userfunc_union(cphvb_userfunc_reduce arg) { return new cphvb_userfunc_union(arg); }
            public static implicit operator cphvb_userfunc_union(cphvb_userfunc_random arg) { return new cphvb_userfunc_union(arg); }
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_reduce
        {
            public cphvb_intp id;
            public cphvb_intp nout;
            public cphvb_intp nin;
            public cphvb_intp struct_size;
            public cphvb_array_ptr operand0;
            public cphvb_array_ptr operand1;
            public cphvb_index axis;
            public cphvb_opcode opcode;

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

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_random
        {
            public cphvb_intp id;
            public cphvb_intp nout;
            public cphvb_intp nin;
            public cphvb_intp struct_size;
            public cphvb_array_ptr operand;

            public cphvb_userfunc_random(cphvb_intp func, cphvb_array_ptr op)
            {
                this.id = func;
                this.nout = 1;
                this.nin = 0;
                this.struct_size = RANDOMFUNC_SIZE;
                this.operand = op;
            }
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_matmul
        {
            public cphvb_intp id;
            public cphvb_intp nout;
            public cphvb_intp nin;
            public cphvb_intp struct_size;
            public cphvb_array_ptr operand0;
            public cphvb_array_ptr operand1;
            public cphvb_array_ptr operand2;

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

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_userfunc_plain
        {
            public cphvb_intp id;
            public cphvb_intp nout;
            public cphvb_intp nin;
            public cphvb_intp struct_size;
            public cphvb_array_ptr operand0;
            public cphvb_array_ptr operand1;
            public cphvb_array_ptr operand2;

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

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 0)]
        public struct cphvb_instruction : IInstruction
        {
            //Instruction status
            public cphvb_error status;
            //Opcode: Identifies the operation
            public cphvb_opcode opcode;
            //Id of each operand, we have explicitly stated them here, instead of using the array style
            public cphvb_array_ptr operand0;
            public cphvb_array_ptr operand1;
            public cphvb_array_ptr operand2;
            //A constant value, used if operand has IntPtr.Zero
            public cphvb_constant constant;     
            //Points to the user-defined function when the opcode is
            //CPHVB_USERFUNC.
            public IntPtr userfunc;

            public cphvb_instruction(cphvb_opcode opcode, cphvb_array_ptr operand, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
            {
                this.status = cphvb_error.CPHVB_INST_UNDONE;
                this.opcode = opcode;
                this.operand0 = operand;
                this.operand1 = cphvb_array_ptr.Null;
                this.operand2 = cphvb_array_ptr.Null;
                this.userfunc = IntPtr.Zero;
                this.constant = constant;
            }

            public cphvb_instruction(cphvb_opcode opcode, cphvb_array_ptr operand1, cphvb_array_ptr operand2, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
            {
                this.status = cphvb_error.CPHVB_INST_UNDONE;
                this.opcode = opcode;
                this.operand0 = operand1;
                this.operand1 = operand2;
                this.operand2 = cphvb_array_ptr.Null;
                this.userfunc = IntPtr.Zero;
                this.constant = constant;
            }

            public cphvb_instruction(cphvb_opcode opcode, cphvb_array_ptr operand1, cphvb_array_ptr operand2, cphvb_array_ptr operand3, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
            {
                this.status = cphvb_error.CPHVB_INST_UNDONE;
                this.opcode = opcode;
                this.operand0 = operand1;
                this.operand1 = operand2;
                this.operand2 = operand3;
                this.userfunc = IntPtr.Zero;
                this.constant = constant;
            }

            public cphvb_instruction(cphvb_opcode opcode, IEnumerable<cphvb_array_ptr> operands, PInvoke.cphvb_constant constant = new PInvoke.cphvb_constant())
            {
                this.status = cphvb_error.CPHVB_INST_UNDONE;
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

            public cphvb_instruction(cphvb_opcode opcode, IntPtr userfunc)
            {
                this.status = cphvb_error.CPHVB_INST_UNDONE;
                this.opcode = opcode;
                this.userfunc = userfunc;
                this.operand0 = cphvb_array_ptr.Null;
                this.operand1 = cphvb_array_ptr.Null;
                this.operand2 = cphvb_array_ptr.Null;
                this.constant = new cphvb_constant();
            }

            public override string ToString()
            {
                return string.Format("{0}({1}, {2}, {3})", this.opcode, operand0, operand1, operand2);
            }

            cphvb_opcode IInstruction.OpCode
            {
                get { return opcode; }
            }
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_init(ref cphvb_component self);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_shutdown();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_execute(cphvb_intp count, [In, Out]cphvb_instruction[] inst_list);
        //[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        //public delegate cphvb_error cphvb_execute_with_userfunc(cphvb_intp count, [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef=typeof(InstructionMarshal))] cphvb_instruction[] inst_list);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_reg_func(string fun, ref cphvb_intp id);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate cphvb_error cphvb_create_array(
                                   cphvb_array_ptr basearray,
                                   cphvb_type     type,
                                   cphvb_intp     ndim,
                                   cphvb_index    start,
                                   cphvb_index[]    shape,
                                   cphvb_index[]    stride,
                                   out cphvb_array_ptr new_array);

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
        /// <param name="count">Number of children components</param>
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
        /// <param name="data">The new data pointer</param>
        /// <returns>Error code (CPHVB_SUCCESS, CPHVB_ERROR)</returns>
        [DllImport("libcphvb", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public extern static cphvb_error cphvb_data_malloc([In] cphvb_array_ptr array);

        /// <summary>
        /// Set the data pointer for the array.
        /// Can only set to non-NULL if the data ptr is already NULL
        /// </summary>
        /// <param name="array">The array in question</param>
        /// <param name="data">The new data pointer</param>
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
