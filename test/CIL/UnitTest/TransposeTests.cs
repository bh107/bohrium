using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Float;
using NumCIL;

namespace UnitTest
{
    public static class TransposeTests
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



        }
    }
}
