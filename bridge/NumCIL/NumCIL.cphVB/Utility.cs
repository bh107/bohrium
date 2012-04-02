using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    public static class Utility
    {
        public static void Activate<T>()
        {
            NumCIL.Generic.NdArray<T>.AccessorFactory = new cphVBAccessorFactory<T>();
        }

        public static void Flush()
        {
            VEM.Instance.Flush();
        }
    }
}
