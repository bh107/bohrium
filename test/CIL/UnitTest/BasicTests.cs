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
using NumCIL.Float;
using NumCIL;
using T = System.Single;

namespace UnitTest
{
    public static class BasicTests
    {
        public static void RunTests()
        {
            var an = Generate.Empty(2000);
            var bn = Generate.Ones(2000);

            var cn = Generate.Empty(2000);
            var dn = Generate.Ones(2000);

            var an0 = an.Value[0];
            var bn0 = bn.Value[0];
            if (bn0 != 1)
                throw new Exception("Error in accessor");

            cn.Value[0] = 1;
            dn.Value[0] = 2;

            var cn0 = cn.Value[0];
            var dn0 = dn.Value[0];

            if (cn0 != 1 || dn0 != 2)
                throw new Exception("Error in accessor");

            var test = Generate.Arange(3) * 4;

            Shape s = new Shape(
                new long[] { 2, 1, 2, 3 },  //Dimension sizes
                6,                          //Offset
                new long[] { 18, 18, 6, 1 } //Strides
            );

            var a = Generate.Arange(s.Length);
            var b = a.Reshape(s);
            var c = b[1][0][1];
            var d = c[2];
            var e = b.Flatten();
            if (e.AsArray().LongLength != 12 || e.AsArray()[3] != 12)
                throw new Exception("Failure in flatten");

            List<T> fln = new List<T>(b[1, 0, 1].Value);
            if (fln.Count != 3) throw new Exception("Failure in basic test");
            if (fln[0] != 30) throw new Exception("Failure in basic test");
            if (fln[1] != 31) throw new Exception("Failure in basic test");
            if (fln[2] != 32) throw new Exception("Failure in basic test");

            T n = b.Value[1, 0, 1, 2];
            if (n != 32) throw new Exception("Failure in basic test");
            if (c.Value[2] != 32) throw new Exception("Failure in basic test");
            if (c.Value[0] != 30) throw new Exception("Failure in basic test");
            if (c.Value[1] != 31) throw new Exception("Failure in basic test");
            if (b.Sum() != 228) throw new Exception("Failure in basic test");

            var r1 = Generate.Arange(12).Reshape(new long[] { 2, 1, 2, 3 });
            if (!Equals(r1.AsArray(), new T[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })) throw new Exception("Failure in basic test");
            var r2 = r1.Reduce<Add>(0);
            if (!Equals(r2.AsArray(), new T[] { 6, 8, 10, 12, 14, 16 })) throw new Exception("Failure in basic test");
            r2 = r1.Reduce<Add>(1);
            if (!Equals(r2.AsArray(), new T[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })) throw new Exception("Failure in basic test");
            r2 = r1.Reduce<Add>(2);
            if (!Equals(r2.AsArray(), new T[] { 3, 5, 7, 15, 17, 19 })) throw new Exception("Failure in basic test");
            r2 = r1.Reduce<Add>(3);
            if (!Equals(r2.AsArray(), new T[] { 3, 12, 21, 30 })) throw new Exception("Failure in basic test");

            var r3 = b.Reduce<Add>();
            if (!Equals(r3.AsArray(), new T[] { 30, 32, 34, 42, 44, 46 })) throw new Exception("Failure in basic test");

            var x1 = Generate.Arange(12).Reshape(new long[] { 4, 3 });
            var x2 = Generate.Arange(3);

            var x3 = x1 + x2;

            var sqrd = x3.Sqrt();
            sqrd *= sqrd;
            sqrd = sqrd.Round();
            sqrd += 4;
            sqrd++;

            if (UFunc.Reduce<T, Add>(UFunc.Reduce<T, Add>(sqrd, 0)).Value[0] != 138) throw new Exception("Failure in basic arithmetics");
            if (UFunc.Reduce<T, Add>(UFunc.Reduce<T, Add>(sqrd, 1)).Value[0] != 138) throw new Exception("Failure in basic arithmetics");

            var x5 = sqrd.Apply((x) => x % 2 == 0 ? x : -x);
            if (!Equals(x5.AsArray(), new T[] { -5, -7, -9, 8, 10, 12, -11, -13, -15, 14, 16, 18 })) throw new Exception("Failure in basic test");

            NumCIL.UFunc.Apply<T, Add>(x1, x2, x3);
            NumCIL.Double.NdArray x4 = (NumCIL.Double.NdArray)x3;
            if (!Equals(x4.AsArray(), new double[] { 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13 })) throw new Exception("Failure in basic test");

            var x6 = Generate.Arange(6).Reshape(new long[] { 2, 3 });

            var x7 = x6.Reduce<Add>();

            var rx7 = x7.AsArray();
            if (!Equals(rx7, new T[] { 3, 5, 7 })) throw new Exception(string.Format("Failure in basic test: [{0}]", string.Join(", ", rx7.Select(x => x.ToString()).ToArray())));

            var x8 = Generate.Arange(10) * 0.5f;
            var rx8 = x8.Reduce<Add>().Value[0];
            if (rx8 != 22.5)
                throw new Exception(string.Format("Failure in broadcast multiply: {0}", rx8));

            var x9 = Mul.Apply(Generate.Arange(10), 0.5f);
            var rx9 = x9.Reduce<Add>().Value[0];
            if (rx9 != 22.5)
                throw new Exception(string.Format("Failure in broadcast multiply: {0}", rx9));

            var x10 = 5 - Generate.Arange(10);
            var x11 = Generate.Arange(10) - 5;
            if (x10.Sum() != 5 || x11.Sum() != -5)
                throw new Exception("Failure in scalar rhs/lhs");

            var n0 = Generate.Arange(4);
            var n1 = n0[new Range(1, 4)];
            var n2 = n0[new Range(0, 3)];
            var n3 = n1 - n2;
            if (n3.Reduce<Add>().Value[0] != 3)
                throw new Exception("Failure in basic slicing");

            var z0 = Generate.Arange(new long[] {2, 2, 3});
            var z1 = z0[Range.All, Range.All, Range.El(1)];
            var z2 = z0[Range.All, Range.El(1), Range.El(1)];
            var z3 = z0[Range.El(1), Range.El(1), Range.All];
            var z4 = z0[Range.El(0), Range.El(1), Range.El(1)];
            var z5 = z0[Range.El(0)];

            if (z1.Shape.Elements != 4)
                throw new Exception("Reduced range failed");
            if (z2.Shape.Elements != 2 || z2.Shape.Dimensions.LongLength != 1)
                throw new Exception("Reduced range failed");
            if (z3.Shape.Elements != 3 || z3.Shape.Dimensions.LongLength != 1)
                throw new Exception("Reduced range failed");
            if (z4.Shape.Elements != 1 || z4.Shape.Dimensions.LongLength != 1)
                throw new Exception("Reduced range failed");
            if (z5.Shape.Elements != 6 || z5.Shape.Dimensions.LongLength != 2)
                throw new Exception("Reduced range failed");

        }

        private static bool Equals(double[] a, double[] b)
        {
            if (a.LongLength != b.LongLength)
                return false;

            for (long i = 0; i < a.LongLength; i++)
                if (a[i] != b[i])
                    return false;

            return true;
        }

        private static bool Equals(float[] a, float[] b)
        {
            if (a.LongLength != b.LongLength)
                return false;

            for (long i = 0; i < a.LongLength; i++)
                if (a[i] != b[i])
                    return false;

            return true;
        }
    }
}
