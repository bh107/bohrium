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
            var e = b.Flatten();
            if (e.Data.LongLength != 12 || e.Data[4] != 12)
                throw new Exception("Failure in flatten");

            List<float> fln = new List<float>(b[1, 0, 1].Value);
            if (fln.Count != 3) throw new Exception("Failure in basic test");
            if (fln[0] != 30) throw new Exception("Failure in basic test");
            if (fln[1] != 31) throw new Exception("Failure in basic test");
            if (fln[2] != 32) throw new Exception("Failure in basic test");

            float n = b.Value[1, 0, 1, 2];
            if (n != 32) throw new Exception("Failure in basic test");
            if (c.Value[2] != 32) throw new Exception("Failure in basic test");
            if (c.Value[0] != 30) throw new Exception("Failure in basic test");
            if (c.Value[1] != 31) throw new Exception("Failure in basic test");

            var r1 = Generate.Arange(12).Reshape(new long[] { 2, 1, 2, 3 });
            if (!Equals(r1.Data, new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })) throw new Exception("Failure in basic test");
            var r2 = r1.Reduce<Add>(0);
            if (!Equals(r2.Data, new float[] { 6, 8, 10, 12, 14, 16 })) throw new Exception("Failure in basic test");
            r2 = r1.Reduce<Add>(1);
            if (!Equals(r2.Data, new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })) throw new Exception("Failure in basic test");
            r2 = r1.Reduce<Add>(2);
            if (!Equals(r2.Data, new float[] { 3, 5, 7, 15, 17, 19 })) throw new Exception("Failure in basic test");
            r2 = r1.Reduce<Add>(3);
            if (!Equals(r2.Data, new float[] { 3, 12, 21, 30 })) throw new Exception("Failure in basic test");

            var r3 = b.Reduce<Add>();
            if (!Equals(r3.Data, new float[] { 30, 32, 34, 42, 44, 46 })) throw new Exception("Failure in basic test");

            var x1 = Generate.Arange(12).Reshape(new long[] { 4, 3 });
            var x2 = Generate.Arange(3);

            var x3 = x1 + x2;

            var sqrd = x3.Sqrt();
            sqrd *= sqrd;
            sqrd = sqrd.Round();
            sqrd += 4;
            sqrd++;

            if (UFunc.Reduce<float, Add>(UFunc.Reduce<float, Add>(sqrd, 0)).Value[0] != 138) throw new Exception();
            if (UFunc.Reduce<float, Add>(UFunc.Reduce<float, Add>(sqrd, 1)).Value[0] != 138) throw new Exception();

            var x5 = sqrd.Apply((x) => x % 2 == 0 ? x : -x);
            if (!Equals(x5.Data, new float[] { -5, -7, -9, 8, 10, 12, -11, -13, -15, 14, 16, 18 })) throw new Exception("Failure in basic test");

            NumCIL.UFunc.Apply<float, Add>(x1, x2, x3);
            NumCIL.Double.NdArray x4 = (NumCIL.Double.NdArray)x3;
            if (!Equals(x4.Data, new double[] { 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13 })) throw new Exception("Failure in basic test");

            var x6 = Generate.Arange(6).Reshape(new long[] { 2, 3 });

            var x7 = x6.Reduce<Add>();

            if (!Equals(x7.Data, new float[] { 3, 5, 7 })) throw new Exception("Failure in basic test");
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
