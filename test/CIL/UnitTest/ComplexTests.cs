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

namespace UnitTest
{
    public static class ComplexTests
    {
        public static void RunTests()
        {
            var a = NumCIL.Complex64.Generate.Range(10);
            var b = NumCIL.Float.Generate.Range(10);
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

            var f = NumCIL.Complex128.Generate.Range(10);
            var g = NumCIL.Double.Generate.Range(10);
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
