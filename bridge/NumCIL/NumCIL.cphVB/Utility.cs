using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    public static class Utility
    {
        public static void SetupDebugEnvironmentVariables()
        {
            try
            {
                var allowednames = new string[] { "cphvb", "cphvb_priv", "cphvb-priv" };
                string basepath = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
                Func<string, bool> eq = (x) =>
                {
                    foreach (var s in allowednames)
                        if (string.Equals(s, x, StringComparison.InvariantCultureIgnoreCase))
                            return true;
                    return false;
                };

                var root = System.IO.Path.GetPathRoot(basepath);
                while (basepath != root && !eq(System.IO.Path.GetFileName(basepath)))
                    basepath = System.IO.Path.GetDirectoryName(basepath);

                if (!eq(System.IO.Path.GetFileName(basepath)))
                    throw new Exception(string.Format("Unable to find a directory named {0}, in path {1}, searched until {2}", "'" + string.Join("', '", allowednames) + "'", System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location), basepath));

                string binary_lookup_path = System.IO.Path.Combine(basepath, "core") + System.IO.Path.PathSeparator + System.IO.Path.Combine(basepath, "vem", "node") + System.IO.Path.PathSeparator;

                //Bad OS detection :)
                if (System.IO.Path.PathSeparator == ':')
                {
                    string ldpath = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
                    string dyldpath = Environment.GetEnvironmentVariable("DYLD_LIBRARY_PATH") ?? "";

                    Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", binary_lookup_path + ldpath);
                    Environment.SetEnvironmentVariable("DYLD_LIBRARY_PATH", binary_lookup_path + dyldpath);

                    bool isOsx = false;
                    try
                    {
                        isOsx = string.Equals("Darwin", System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo("uname") { UseShellExecute = false, RedirectStandardOutput = true }).StandardOutput.ReadToEnd());
                    }
                    catch { }

                    if (isOsx)
                        Environment.SetEnvironmentVariable("CPHVB_CONFIG", System.IO.Path.Combine(basepath, "config.osx.ini"));
                    else
                        Environment.SetEnvironmentVariable("CPHVB_CONFIG", System.IO.Path.Combine(basepath, "config.ini"));
                }
                else
                {
                    binary_lookup_path += System.IO.Path.Combine(basepath, "pthread_win32");
                    string path = Environment.GetEnvironmentVariable("PATH");
                    Environment.SetEnvironmentVariable("PATH", path + System.IO.Path.PathSeparator + binary_lookup_path);
                    Environment.SetEnvironmentVariable("CPHVB_CONFIG", System.IO.Path.Combine(basepath, "config.win.ini"));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to set up debug paths for cphVB: " + ex.ToString());
            }
        }

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
