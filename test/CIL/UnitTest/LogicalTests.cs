using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace UnitTest
{
	public static class LogicalTests
	{
		public static void RunTests()
		{
			var a = NumCIL.Float.Generate.Arange(10);
			var b = NumCIL.Float.Generate.Ones(10);
			var c = a > 5;
			var d = 5 < a;
			var e = a == b;
			var f = (NumCIL.UInt8.NdArray)c;
			var g = (NumCIL.UInt8.NdArray)d;
			var h = (NumCIL.UInt8.NdArray)e;

			if (f.Sum() != 4)
				throw new Exception("Error in compare");
			if (g.Sum() != 4)
				throw new Exception("Error in compare");
			if (h.Sum() != 1)
				throw new Exception("Error in compare");

			var i = (NumCIL.UInt8.NdArray)(c ^ !e);
			var j = (NumCIL.UInt8.NdArray)(c & !e);
			var k = (NumCIL.UInt8.NdArray)(c | e);
			if (i.Sum() != 5 || j.Sum() != 4 || k.Sum() != 5)
				throw new Exception("Error in logical operation");

			var l = (NumCIL.UInt8.NdArray)((a >= 4) | (a <= 2));
			if (l.Sum() != 9)
				throw new Exception("Error in compare");

		}
	}
}

