using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    public static class Utility
    {
        public static void Activate()
        {
            //Activate the instance so timings are more accurate when profiling
            //and also ensure that config problems are found during startup
            VEM.Instance.Flush();
            Activate<float>();
            Activate<double>();
            Activate<sbyte>();
            Activate<short>();
            Activate<int>();
            Activate<long>();
            Activate<byte>();
            Activate<ushort>();
            Activate<uint>();
            Activate<ulong>();

        }

        public static void Deactivate()
        {
            Deactivate<float>();
            Deactivate<double>();
            Deactivate<sbyte>();
            Deactivate<short>();
            Deactivate<int>();
            Deactivate<long>();
            Deactivate<byte>();
            Deactivate<ushort>();
            Deactivate<uint>();
            Deactivate<ulong>();
        }

        public static void Activate<T>()
        {
            NumCIL.Generic.NdArray<T>.AccessorFactory = new cphVBAccessorFactory<T>();
        }

        public static void Deactivate<T>()
        {
            NumCIL.Generic.NdArray<T>.AccessorFactory = new NumCIL.Generic.DefaultAccessorFactory<T>();
        }

        public static void Flush()
        {
            VEM.Instance.Flush();
        }
    }
}
