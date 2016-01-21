using System;
using System.Collections.Generic;
using System.Linq;

using NumCIL.Generic.Operators;
using NumCIL.Generic;

namespace NumCIL.Bohrium
{
	internal static partial class NameMapper
	{
		private static readonly KeyValuePair<Type, string>[] optypemap = 
		{
			new KeyValuePair<Type, string>(typeof(IAdd), "add"),
			new KeyValuePair<Type, string>(typeof(ISub), "subtract"),
			new KeyValuePair<Type, string>(typeof(IMul), "multiply"),
			new KeyValuePair<Type, string>(typeof(IDiv), "divide"),
			new KeyValuePair<Type, string>(typeof(IPow), "power"),
			new KeyValuePair<Type, string>(typeof(IAbs), "absolute"),
			new KeyValuePair<Type, string>(typeof(IGreaterThan), "greater"),
			new KeyValuePair<Type, string>(typeof(IGreaterThanOrEqual), "greater_equal"),
			new KeyValuePair<Type, string>(typeof(ILessThan), "less"),
			new KeyValuePair<Type, string>(typeof(ILessThanOrEqual), "less_equal"),
			new KeyValuePair<Type, string>(typeof(IEqual), "equal"),
			new KeyValuePair<Type, string>(typeof(INotEqual), "not_equal"),
			new KeyValuePair<Type, string>(typeof(IAnd), "bitwise_and"),
			new KeyValuePair<Type, string>(typeof(IOr), "bitwise_or"),
			new KeyValuePair<Type, string>(typeof(IXor), "bitwise_xor"),
			new KeyValuePair<Type, string>(typeof(INot), "logical_not"),
			new KeyValuePair<Type, string>(typeof(IInvert), "invert"),
			new KeyValuePair<Type, string>(typeof(IMax), "maximum"),
			new KeyValuePair<Type, string>(typeof(IMin), "minimum"),
			//new KeyValuePair<Type, string>(typeof(IShiftLeft), "left_shift"),
			//new KeyValuePair<Type, string>(typeof(IShiftRight), "right_shift"),

			new KeyValuePair<Type, string>(typeof(ISin), "sin"),
			new KeyValuePair<Type, string>(typeof(ICos), "cos"),
			new KeyValuePair<Type, string>(typeof(ITan), "tan"),
			new KeyValuePair<Type, string>(typeof(ISinh), "sinh"),
			new KeyValuePair<Type, string>(typeof(ICosh), "cosh"),
			new KeyValuePair<Type, string>(typeof(ITanh), "tanh"),
			new KeyValuePair<Type, string>(typeof(IAsin), "arcsin"),
			new KeyValuePair<Type, string>(typeof(IAcos), "arccos"),
			new KeyValuePair<Type, string>(typeof(IAtan), "arctan"),
			//new KeyValuePair<Type, string>(typeof(IAsinh), "arcsinh"),
			//new KeyValuePair<Type, string>(typeof(IAcosh), "arccosh"),
			//new KeyValuePair<Type, string>(typeof(IAtanh), "arctanh"),
			//new KeyValuePair<Type, string>(typeof(IAtan2), "arctan2"),

			new KeyValuePair<Type, string>(typeof(IExp), "exp"),
			//new KeyValuePair<Type, string>(typeof(IExp2), "exp2"),
			//new KeyValuePair<Type, string>(typeof(IExpm1), "expm1"),
			new KeyValuePair<Type, string>(typeof(ILog), "log"),
			//new KeyValuePair<Type, string>(typeof(ILog2), "log2"),
			new KeyValuePair<Type, string>(typeof(ILog10), "log10"),
			//new KeyValuePair<Type, string>(typeof(ILog1p), "log1p"),
			new KeyValuePair<Type, string>(typeof(ISqrt), "sqrt"),
			new KeyValuePair<Type, string>(typeof(ICeiling), "ceil"),
			//new KeyValuePair<Type, string>(typeof(ITruncate), "trunc"),
			new KeyValuePair<Type, string>(typeof(IFloor), "floor"),
			new KeyValuePair<Type, string>(typeof(IRound), "rint"),
			new KeyValuePair<Type, string>(typeof(IMod), "mod"),
			//new KeyValuePair<Type, string>(typeof(IIsNan), "isnan"),
			//new KeyValuePair<Type, string>(typeof(IIsInf), "isinf"),
			new KeyValuePair<Type, string>(typeof(ICopyOperation), "identity"),
			new KeyValuePair<Type, string>(typeof(ITypeConversion), "identity"),
			
			new KeyValuePair<Type, string>(typeof(IRealValue), "real"),
			new KeyValuePair<Type, string>(typeof(IImaginaryValue), "imag"),
			new KeyValuePair<Type, string>(typeof(ISign), "sign"),
		};

		private static readonly Dictionary<Type, Type> pointertype_map;
		private static readonly Dictionary<Type, string> typename_map;
		private static readonly Dictionary<Type, string> opname_map;
		private static readonly Dictionary<Type, string> opname_bool_map;

		static NameMapper()
		{
			pointertype_map = pointertypemap.ToDictionary(x => x.Key, x => x.Value);
			typename_map = typenamemap.ToDictionary(x => x.Key, x => x.Value);
			opname_map = optypemap.ToDictionary(x => x.Key, x => x.Value);
			opname_bool_map = optypemap.ToDictionary(x => x.Key, x => x.Value);
			opname_bool_map[typeof(IAnd)] = "logical_and";
			opname_bool_map[typeof(IOr)] = "logical_or";
			opname_bool_map[typeof(IXor)] = "logical_xor";
		}

		public static string GetOperationName<T, C>()
		{
			return GetOperationName(typeof(T), typeof(C));
		}

		public static string GetOperationName<T>(Type op)
		{
			return GetOperationName(typeof(T), op);
		}

		public static string GetOperationName(Type t, Type op)
		{
			string res;
			if ((op == typeof(bool) ? opname_bool_map : opname_map).TryGetValue(t, out res))
				return res;

			foreach(var ti in t.GetInterfaces())
				if ((op == typeof(bool) ? opname_bool_map : opname_map).TryGetValue(ti, out res))
					return res;
			
			return t.FullName;
		}

		public static string GetTypeName<T>()
		{
			return GetTypeName(typeof(T));
		}

		public static string GetTypeName(Type t)
		{
			string res;
			if (typename_map.TryGetValue(t, out res))
				return res;
			return t.FullName;
		}

		public static Type GetPointerType<T>()
		{
			return GetPointerType(typeof(T));
		}

		public static Type GetPointerType(Type t)
		{
			Type res;
			if (pointertype_map.TryGetValue(t, out res))
				return res;
			return t;
		}

		public static IMultiArray WrapWithPointer<T>(this NdArray<T> data)
		{
			var ptype = GetPointerType<T>();
			if (ptype == typeof(T))
				throw new Exception(string.Format("Unable to get Bohrium pointer type for {0}", typeof(T).FullName));

			return (IMultiArray)Activator.CreateInstance(ptype, data);
		}

	}
}

