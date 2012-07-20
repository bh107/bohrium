using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace UnitTest
{
    public static class ComplexTests
    {
        public static void RunTests()
        {
            var a = NumCIL.Complex64.Generate.Arange(10);
            var b = NumCIL.Float.Generate.Arange(10);
            var c = a * b; //Testing implicit type conversion
			var cx = c.Sum().Real;
            var d = c.Real();
            var e = d.Sum();
            if (e != 285 || cx != 285)
                throw new Exception("Error in complex64");

            //Testing scalar
            var zx = (c * 2);
            zx.Flush();
            var xy = zx.Real();
            if (xy.Sum() != 570)
                throw new Exception("Error in complex64");

            var f = NumCIL.Complex128.Generate.Arange(10);
            var g = NumCIL.Double.Generate.Arange(10);
            var h = f * g; //Testing implicit type conversion
			var hx = h.Sum().Real;
            var i = h.Real();
            var j = i.Sum();
            if (j != 285 || hx != 285)
                throw new Exception("Error in complex128");

            //Testing scalar
            if ((h * 2).Real().Sum() != 570)
                throw new Exception("Error in complex128");
        }
    }
}
