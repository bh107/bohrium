using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Float;
using NumCIL;

namespace UnitTest
{
    public static class ExtendedTests
    {
        public static void RunTests()
        {
            var a = Generate.Arange(2, 3, 4);
            var b = a.Transpose();

            if (b.Value[0, 0, 1] != 12)
                throw new Exception("Something failed in transpose");

            var c = Generate.Zeroes(100);
            c[new Range(0, 0, (long)Math.Sqrt(c.Shape.Length) + 1)] = 1;

            if (c.Value[99] != 1 || c.Value[22] != 1 || c.Value[10] != 0 || c.Value[12] != 0 || Add.Reduce(c).Value[0] != 10)
                throw new Exception("Something failed in stride tricks");


            var d = Generate.Arange(8) + 1;
            var e = d.Repeat(2);
            if (e.Data.LongLength != 16 || Add.Reduce(e).Value[0] != 72)
                throw new Exception("Something failed in simple repeat");
            var f = d.Repeat(3);
            if (f.Data.LongLength != 24 || Add.Reduce(f).Value[0] != 108)
                throw new Exception("Something failed in simple repeat");

            var g = d.Repeat(new long[] { 1, 2, 1, 3, 4, 1, 1, 2 });
            if (g.Data.LongLength != 15 || Add.Reduce(g).Value[0] != 69)
                throw new Exception("Something failed in extended repeat");

            var h = d.Reshape(new Shape(new long[] { 2, 2, 2 }));
            var i = h.Repeat(4, 0);
            if (i.Shape.Dimensions[0].Length != 8 || i.Shape.Elements != 8*4 || Add.Reduce(Add.Reduce(Add.Reduce(i))).Value[0] != 144)
                throw new Exception("Something failed in axis repeat");

            var j = h.Repeat(4, 1);
            if (j.Shape.Dimensions[1].Length != 8 || j.Shape.Elements != 8*4 || Add.Reduce(Add.Reduce(Add.Reduce(j))).Value[0] != 144)
                throw new Exception("Something failed in axis repeat");

            var k = h.Repeat(4, 2);
            if (k.Shape.Dimensions[2].Length != 8 || k.Shape.Elements != 8 * 4 || Add.Reduce(Add.Reduce(Add.Reduce(k))).Value[0] != 144)
                throw new Exception("Something failed in axis repeat");

            var l = h.Repeat(new long[] {4, 2}, 0);
            if (l.Shape.Dimensions[0].Length != 6 || l.Shape.Elements != 24 || Add.Reduce(Add.Reduce(Add.Reduce(l))).Value[0] != 92)
                throw new Exception("Something failed in axis repeat");

            var m = h.Repeat(new long[] { 4, 2 }, 1);
            if (m.Shape.Dimensions[1].Length != 6 || m.Shape.Elements != 24 || Add.Reduce(Add.Reduce(Add.Reduce(m))).Value[0] != 100)
                throw new Exception("Something failed in axis repeat");

            var n = h.Repeat(new long[] { 4, 2 }, 2);
            if (n.Shape.Dimensions[2].Length != 6 || n.Shape.Elements != 24 || Add.Reduce(Add.Reduce(Add.Reduce(n))).Value[0] != 104)
                throw new Exception("Something failed in axis repeat");

            var o = n.Flatten();
            if (o.Data.LongLength != 24)
                throw new Exception("Something failed in flatten");

            if (o.Sum() != 104 || o.Max() != 8)
                throw new Exception("Something failed in aggregate");
        }
    }
}
