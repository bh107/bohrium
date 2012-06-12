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
            a = a.Negate();
            a = Inc.Apply(a);
            a = Abs.Apply(a);

            a = a.Reduce(Ops.Add);
            a = Add.Reduce(a);

            if (a.Value[0] != 48)
                throw new Exception("Something went wrong");
        }
    }
}
