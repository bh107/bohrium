using System;
using System.Collections.Generic;
using System.Linq;

namespace Utilities
{
	public static class RunBenchmark
	{
		public struct InstanceData
		{
			public long[] sizes;
			public bool use_bohrium;
			public System.Type type;

			public Dictionary<string, string> cmd_opts;
		}
	
		public static void Run(string[] args, long size_count, Action<InstanceData> run, Action<InstanceData> validate = null)
		{
			Dictionary<string, string> dict = CommandLineParser.ExtractOptions(args.ToList());
			if (!dict.ContainsKey("size"))
				throw new ArgumentException(string.Format("Main() needs the size argument (e.g --size=1000*1000{0})", size_count == 3 ? "*10" : ""));

			string dtype;
			if (!dict.TryGetValue("dtype", out dtype))
				dtype = "float";
			
			var sizes = (from n in dict["size"].Split('*') select Convert.ToInt64(n)).ToArray();
			if (sizes.Length != size_count)
				throw new ArgumentException(string.Format("The size argument must consist of {0} dimensions (e.g. --size=1000*1000{1})", size_count, size_count == 3 ? "*10" : ""));
			
			var use_bohrium = ParseBoolOption(dict, "bohrium", true);
			
			var t = typeof(float);
			switch (dtype.ToLowerInvariant())
			{
				case "double":
					t = typeof(double);
					break;
				case "single":
				case "float":
					t = typeof(float);
					break;
				case "byte":
				case "char":
				case "uint8":
					t = typeof(byte);
					break;
				case "int8":
				case "sbyte":
					t = typeof(sbyte);
					break;
				case "int16":
					t = typeof(short);
					break;
				case "uint16":
					t = typeof(ushort);
					break;
				case "int32":
					t = typeof(int);
					break;
				case "uint32":
					t = typeof(uint);
					break;
				case "int64":
					t = typeof(long);
					break;
				case "uint64":
					t = typeof(ulong);
					break;
				default:
					throw new ArgumentException(string.Format("Bad input type: {0}", dtype));
			}
			
			var data = new InstanceData() { sizes = sizes, type = t, use_bohrium = use_bohrium, cmd_opts = dict };
			
			if (validate != null)
				validate(data);
			
			if (use_bohrium)
			{
				try
				{
					NumCIL.Bohrium.Utility.Activate();
				}
				catch(Exception ex)
				{
					data.use_bohrium = false;
					use_bohrium = false;

					try
					{
						NumCIL.Bohrium.Utility.Deactivate();
					}
					catch
					{
					}

					if (dict.ContainsKey("bohrium"))
						throw ex;
					else
						Console.WriteLine("Bohrium failed to load, running without.{0}Error message: {1}", Environment.NewLine, ex.Message);
				}
			}
				
			run(data);
			
			if (use_bohrium)
				NumCIL.Bohrium.Utility.Deactivate();
		}
		
		public static bool ParseBoolOption(IDictionary<string, string> opts, string name, bool @default = false)
		{
			string s;
			opts.TryGetValue(name, out s);
			
			if (new string[] {"t", "true", "1", "on", "yes" }.Any(f => f.Equals(s, StringComparison.InvariantCultureIgnoreCase)))
				return true;
			if (new string[] {"f", "false", "0", "off", "no" }.Any(f => f.Equals(s, StringComparison.InvariantCultureIgnoreCase)))
				return false;

			return @default;
		}
	}
}

