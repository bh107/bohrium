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
                throw new Exception(string.Format("Something failed in stride tricks: {0}, {1}, {2}, {3}, {4}", c.Value[99], c.Value[22], c.Value[10], c.Value[12], Add.Reduce(c).Value[0]));


            var d = Generate.Arange(8) + 1;
            var e = d.Repeat(2);
            if (e.AsArray().LongLength != 16 || Add.Reduce(e).Value[0] != 72)
                throw new Exception("Something failed in simple repeat");
            var f = d.Repeat(3);
            if (f.AsArray().LongLength != 24 || Add.Reduce(f).Value[0] != 108)
                throw new Exception("Something failed in simple repeat");

            var g = d.Repeat(new long[] { 1, 2, 1, 3, 4, 1, 1, 2 });
            if (g.AsArray().LongLength != 15 || Add.Reduce(g).Value[0] != 69)
                throw new Exception("Something failed in extended repeat");

            var h = d.Reshape(new Shape(new long[] { 2, 2, 2 }));
            var i = h.Repeat(4, 0);
            if (i.Shape.Dimensions[0].Length != 8 || i.Shape.Elements != 8 * 4 || Add.Reduce(Add.Reduce(Add.Reduce(i))).Value[0] != 144 || i.Value[4, 0, 0] != 5 || i.Value[4, 1, 1] != 8 || i.Value[3, 0, 0] != 1)
                throw new Exception("Something failed in axis repeat");

            var j = h.Repeat(4, 1);
            if (j.Shape.Dimensions[1].Length != 8 || j.Shape.Elements != 8 * 4 || Add.Reduce(Add.Reduce(Add.Reduce(j))).Value[0] != 144 || j.Value[0, 4, 0] != 3 || j.Value[1, 4, 1] != 8 || j.Value[0, 3, 0] != 1)
                throw new Exception("Something failed in axis repeat");

            var k = h.Repeat(4, 2);
            if (k.Shape.Dimensions[2].Length != 8 || k.Shape.Elements != 8 * 4 || Add.Reduce(Add.Reduce(Add.Reduce(k))).Value[0] != 144 || k.Value[0, 0, 4] != 2 || k.Value[1, 1, 4] != 8 || k.Value[0, 0, 3] != 1)
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
            if (o.AsArray().LongLength != 24)
                throw new Exception("Something failed in flatten");

            if (o.Sum() != 104 || o.Max() != 8)
                throw new Exception("Something failed in aggregate");

            var p = new NdArray(new float[] { 1, 2, 3, 4 }, new long[] { 2, 2 });
            var q = new NdArray(new float[] { 5, 6 }, new long[] { 1, 2 });
            var r = p.Concatenate(q);
            var s = p.Concatenate(q.Transposed, 1);

            if (r.Value[2, 1] != 6)
                throw new Exception("Something failed in Concatenate");
            if (s.Value[1, 2] != 6)
                throw new Exception("Something failed in Concatenate");

            var t = new NdArray(42).Subview(Range.NewAxis, 0);
            var u = o[Range.NewAxis, Range.All].Concatenate(t, 1);
            if (u.Reduce<Add>().Reduce<Add>().Value[0] != 104 + 42)
                throw new Exception("Something failed in Concatenate");

            var v = o.Concatenate(new NdArray(42), 1);
            if (v.Shape.Dimensions.LongLength != 1 || v.Reduce<Add>().Value[0] != 104 + 42)
                throw new Exception("Something failed in broadcast extended Concatenate");

            var w = Generate.Arange(4, 2);
            var x = Generate.Arange(2, 3);
            var y = w.MatrixMultiply(x);
            if (y.Sum() != 228)
                throw new Exception("Failure in matrix multiply");

            var z = w.MatrixMultiply(new NdArray(new float[] {1,2}).Reshape(new Shape(new long[] {2})));
            if (z.Sum() != 44)
                throw new Exception("Failure in matrix multiply");

            var x0 = Generate.Arange(20).Repeat(new long[] { 10 }, 0);
            var x1 = Generate.Arange(20).Repeat(10, 0);
            if (x0.Sum() != 1900 || x1.Sum() != 1900)
                throw new Exception("Failure in repeat");
        }
    }
}
