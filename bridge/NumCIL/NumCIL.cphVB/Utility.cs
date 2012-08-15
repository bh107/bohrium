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

namespace NumCIL.cphVB
{
    /// <summary>
    /// Utility class for cphVB
    /// </summary>
    public static class Utility
    {
        /// <summary>
        /// Attempts to set up cphVB by looking for the cphvb checkout folder.
        /// This simplifies using cphVB directly from the build folder,
        /// without installing cphVB first
        /// </summary>
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
                {
                    basepath = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
                    while (basepath != root && !System.IO.Directory.EnumerateFiles(basepath, "build.py").Any())
                        basepath = System.IO.Path.GetDirectoryName(basepath);

                    if (!System.IO.Directory.EnumerateFiles(basepath, "build.py").Any())
                        throw new Exception(string.Format("Unable to find a directory named {0}, in path {1}, searched until {2}", "'" + string.Join("', '", allowednames) + "'", System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location), basepath));
                }


                string binary_lookup_path = System.IO.Path.Combine(basepath, "core") + System.IO.Path.PathSeparator;

                //Bad OS detection :)
                if (System.IO.Path.PathSeparator == ':')
                {
                    bool isOsx = false;
                    try
                    {
                        isOsx = string.Equals("Darwin", System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo("uname") { UseShellExecute = false, RedirectStandardOutput = true }).StandardOutput.ReadToEnd().Trim());
                    }
                    catch { }

					string configpath = Environment.GetEnvironmentVariable("CPHVB_CONFIG") ?? "";
					if (string.IsNullOrEmpty(configpath))
					{
	                    if (isOsx)
						{
		                    string dyldpath = Environment.GetEnvironmentVariable("DYLD_LIBRARY_PATH") ?? "";
		                    Environment.SetEnvironmentVariable("DYLD_LIBRARY_PATH", binary_lookup_path + dyldpath);
	                        Environment.SetEnvironmentVariable("CPHVB_CONFIG", System.IO.Path.Combine(basepath, "config.osx.ini"));
						}
	                    else
						{
		                    string ldpath = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
		                    Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", binary_lookup_path + ldpath);
	                        Environment.SetEnvironmentVariable("CPHVB_CONFIG", System.IO.Path.Combine(basepath, "config.ini"));
						}
					}
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

        /// <summary>
        /// Activates cphVB for all supported datatypes
        /// </summary>
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
			Activate<bool>();
			Activate<NumCIL.Complex64.DataType>();
			Activate<System.Numerics.Complex>();
        }

        /// <summary>
        /// Deactivates cphVB for all supported datatypes
        /// </summary>
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
			Deactivate<bool>();
			Deactivate<NumCIL.Complex64.DataType>();
			Deactivate<System.Numerics.Complex>();
        }
        
        /// <summary>
        /// Activates cphVB for a specific datatype
        /// </summary>
        /// <typeparam name="T">The datatype to activate cphVB for</typeparam>
		public static void Activate<T>()
        {
            NumCIL.Generic.NdArray<T>.AccessorFactory = new cphVBAccessorFactory<T>();
        }

        /// <summary>
        /// Deactivates cphVB for a specific datatype
        /// </summary>
        /// <typeparam name="T">The datatype to deactivate cphVB for</typeparam>
        public static void Deactivate<T>()
        {
            NumCIL.Generic.NdArray<T>.AccessorFactory = new NumCIL.Generic.DefaultAccessorFactory<T>();
        }

        /// <summary>
        /// Flushes pending operations in the VEM, note that this does not flush all pending instructions
        /// </summary>
        public static void Flush()
        {
            VEM.Instance.Flush();
        }
    }
}
