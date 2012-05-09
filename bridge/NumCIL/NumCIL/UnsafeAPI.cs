using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.IO;
using NumCIL.Generic;

namespace NumCIL
{
    /// <summary>
    /// This class wraps access to the unsafe methods
    /// </summary>
    internal static class UnsafeAPI
    {
        /// <summary>
        /// Gets or sets a value indicating if use of unsafe methods is disabled
        /// </summary>
        public static bool DisableUnsafeAPI { get; set; }
        /// <summary>
        /// Gets a value indicating if unsafe operations are supported by the runtime/environment
        /// </summary>
        public static readonly bool IsUnsafeSupported;
        /// <summary>
        /// The method used to retrieve binary apply methods
        /// </summary>
        private static readonly MethodInfo _getBinaryApply;
        /// <summary>
        /// The method used to retrieve unary apply methods
        /// </summary>
        private static readonly MethodInfo _getUnaryApply;
        /// <summary>
        /// The method used to retrieve nullary apply methods
        /// </summary>
        private static readonly MethodInfo _getNullaryApply;
        /// <summary>
        /// The method used to retrieve aggregate methods
        /// </summary>
        private static readonly MethodInfo _getAggregate;

        /// <summary>
        /// Cache of compiled unsafe binary methods bound to a particular operator
        /// </summary>
        private static readonly Dictionary<object, MethodInfo> _binaryMethodCache = new Dictionary<object, MethodInfo>();
        /// <summary>
        /// Cache of compiled unsafe unary methods bound to a particular operator
        /// </summary>
        private static readonly Dictionary<object, MethodInfo> _unaryMethodCache = new Dictionary<object, MethodInfo>();
        /// <summary>
        /// Cache of compiled unsafe nullary methods bound to a particular operator
        /// </summary>
        private static readonly Dictionary<object, MethodInfo> _nullaryMethodCache = new Dictionary<object, MethodInfo>();
        /// <summary>
        /// Cache of compiled unsafe aggregate methods bound to a particular operator
        /// </summary>
        private static readonly Dictionary<object, MethodInfo> _aggregateMethodCache = new Dictionary<object, MethodInfo>();

        /// <summary>
        /// Static constructor, loads the unsafe dll and probes for support,
        /// and sets up the accessor methods
        /// </summary>
        static UnsafeAPI()
        {
            bool unsafeSupported = false;
            MethodInfo supportsBinaryApply = null;
            MethodInfo supportsUnaryApply = null;
            MethodInfo supportsNullaryApply = null;
            MethodInfo supportsAggregate = null;

            try
            {
                string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "NumCIL.Unsafe.dll");
                Assembly asm = Assembly.LoadFrom(path);
                Type utilityType = asm.GetType("NumCIL.Unsafe.Utility");
                unsafeSupported = (bool)utilityType.GetProperty("SupportsUnsafe").GetValue(null, null);

                if (unsafeSupported)
                {
                    supportsBinaryApply = utilityType.GetMethod("GetBinaryApply");
                    supportsUnaryApply = utilityType.GetMethod("GetUnaryApply");
                    supportsNullaryApply = utilityType.GetMethod("GetNullaryApply");
                    supportsAggregate = utilityType.GetMethod("GetAggregate");
                    unsafeSupported &= supportsBinaryApply != null;
                    unsafeSupported &= supportsUnaryApply != null;
                    unsafeSupported &= supportsNullaryApply != null;
                    unsafeSupported &= supportsAggregate != null;
                }
            }
            catch
            {
                unsafeSupported = false;
            }

            IsUnsafeSupported = unsafeSupported;

            if (IsUnsafeSupported)
            {
                _getBinaryApply = supportsBinaryApply;
                _getUnaryApply = supportsUnaryApply;
                _getNullaryApply = supportsNullaryApply;
                _getAggregate = supportsAggregate;
            }
            else
            {
                _getBinaryApply = null;
                _getUnaryApply = null;
                _getNullaryApply = null;
                _getAggregate = null;
            }

            if (Environment.GetEnvironmentVariable("NUMCIL_DISABLE_UNSAFE") != null)
                DisableUnsafeAPI = true;
        }

        /// <summary>
        /// Attempts to load an unsafe method, compile it for the given type and operator and then execute it with the gicen operands
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side argument</param>
        /// <param name="in2">The right-handside argument</param>
        /// <param name="out">The output target</param>
        /// <returns>True if the operation was supported and executed, false otherwise</returns>
        public static bool UFunc_Op_Inner_Binary_Flush_Unsafe<T, C>(C op, NdArray<T> in1, NdArray<T> in2, ref NdArray<T> @out)
            where C : struct, IBinaryOp<T>
        {
            if (DisableUnsafeAPI || !IsUnsafeSupported)
                return false;

            MethodInfo mi;
            if (!_binaryMethodCache.TryGetValue(op, out mi))
            {
                MethodInfo mix = (MethodInfo)_getBinaryApply.MakeGenericMethod(typeof(T)).Invoke(null, null);
                if (mix != null)
                    mi = mix.MakeGenericMethod(typeof(C));
                
                _binaryMethodCache[op] = mi;
            }

            if (mi == null)
                return false;

            mi.Invoke(null, new object[] { op, in1, in2, @out });
            return true;
        }

        /// <summary>
        /// Attempts to load an unsafe method, compile it for the given type and operator and then execute it with the gicen operands
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The input argument</param>
        /// <param name="out">The output target</param>
        /// <returns>True if the operation was supported and executed, false otherwise</returns>
        public static bool UFunc_Op_Inner_Unary_Flush_Unsafe<T, C>(C op, NdArray<T> in1, ref NdArray<T> @out)
            where C : struct, IUnaryOp<T>
        {
            if (DisableUnsafeAPI || !IsUnsafeSupported)
                return false;

            MethodInfo mi;
            if (!_unaryMethodCache.TryGetValue(op, out mi))
            {
                MethodInfo mix = (MethodInfo)_getUnaryApply.MakeGenericMethod(typeof(T)).Invoke(null, null);
                if (mix != null)
                    mi = mix.MakeGenericMethod(typeof(C));

                _unaryMethodCache[op] = mi;
            }

            if (mi == null)
                return false;

            mi.Invoke(null, new object[] { op, in1, @out });
            return true;
        }

        /// <summary>
        /// Attempts to load an unsafe method, compile it for the given type and operator and then execute it with the gicen operands
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="out">The output target</param>
        /// <returns>True if the operation was supported and executed, false otherwise</returns>
        public static bool UFunc_Op_Inner_Nullary_Flush_Unsafe<T, C>(C op, NdArray<T> @out)
            where C : struct, INullaryOp<T>
        {
            if (DisableUnsafeAPI || !IsUnsafeSupported)
                return false;

            MethodInfo mi;
            if (!_nullaryMethodCache.TryGetValue(op, out mi))
            {
                MethodInfo mix = (MethodInfo)_getNullaryApply.MakeGenericMethod(typeof(T)).Invoke(null, null);
                if (mix != null)
                    mi = mix.MakeGenericMethod(typeof(C));

                _nullaryMethodCache[op] = mi;
            }

            if (mi == null)
                return false;

            mi.Invoke(null, new object[] { op, @out });
            return true;
        }

        /// <summary>
        /// Attempts to load an unsafe method, compile it for the given type and operator and then execute it with the gicen operands
        /// </summary>
        /// <typeparam name="T">The type of data to operate on</typeparam>
        /// <typeparam name="C">The type of operation to perform</typeparam>
        /// <param name="op">The operation instance</param>
        /// <param name="in1">The left-hand-side argument</param>
        /// <param name="res">The output result</param>
        /// <returns>True if the operation was supported and executed, false otherwise</returns>
        public static bool Aggregate_Entry_Unsafe<T, C>(C op, NdArray<T> in1, out T res)
            where C : struct, IBinaryOp<T>
        {
            if (DisableUnsafeAPI || !IsUnsafeSupported)
            {
                res = default(T);
                return false;
            }

            MethodInfo mi;
            if (!_aggregateMethodCache.TryGetValue(op, out mi))
            {
                MethodInfo mix = (MethodInfo)_getAggregate.MakeGenericMethod(typeof(T)).Invoke(null, null);
                if (mix != null)
                    mi = mix.MakeGenericMethod(typeof(C));

                _aggregateMethodCache[op] = mi;
            }

            if (mi == null)
            {
                res = default(T);
                return false;
            }

            res = (T)mi.Invoke(null, new object[] { op, in1 });
            return true;
        }
    }
}
