using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Float;
using NumCIL;

namespace UnitTest
{
    public static class BasicTests
    {
        public static void RunTests()
        {
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

            foreach (var v in b[1, 0, 1].Value)
                Console.WriteLine("{0}", v);

            float n = b.Value[1, 0, 1, 2];
            n = c.Value[2];
            n = d.Value[0];
            n = b.Value[1];

            var r1 = Generate.Arange(12).Reshape(new long[] { 2, 1, 2, 3 });
            var r2 = r1.Reduce<Add>(0);
            r2 = r1.Reduce<Add>(1);
            r2 = r1.Reduce<Add>(2);
            r2 = r1.Reduce<Add>(3);

            var r3 = b.Reduce<Add>();

            var x1 = Generate.Arange(12).Reshape(new long[] { 4, 3 });
            var x2 = Generate.Arange(3);

            var x3 = x1 + x2;

            var sqrd = x3.Sqrt();
            sqrd *= sqrd;
            sqrd = sqrd.Round();
            sqrd += 4;
            sqrd++;

            var x5 = sqrd.Apply((x) => x % 2 == 0 ? x : -x);

            NumCIL.UFunc.Apply<float, Add>(x1, x2, x3);
            NumCIL.Double.NdArray x4 = (NumCIL.Double.NdArray)x3;

            var x6 = Generate.Arange(6).Reshape(new long[] { 2, 3 });

            var x7 = x6.Reduce<Add>();


            Console.WriteLine(x7);
        }
    }
}
