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
using NumCIL.Float;

namespace UnitTest
{
    public static class TypeLookupTests
    {
        private struct GenerateOp : NumCIL.INullaryOp<float>
        {
            public float Op()
            {
                return 1;
            }
        }

        public static void RunTests()
        {
            var a = Generate.Empty(4, 4);
            a.Apply(new GenerateOp());
            a += 1;
            a = a.Apply(Ops.Add, new NdArray(1));
            a = a.Add(1);
            a = a.Sub(new NdArray(1));
            a = Add.Apply(a, 1);
            a = -a;
            a = ++a;
            a = Abs.Apply(a);

            a = a.Reduce(Ops.Add);
            a = Add.Reduce(a);

            if (a.Value[0] != 48)
                throw new Exception("Something went wrong");
        }
    }
}
