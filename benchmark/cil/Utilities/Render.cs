using System;
using System.Drawing;

namespace Utilities
{
	public static class Render
	{
		public static Func<double, Color> ColorMap(Color basecolor)
		{
			return (v) => Color.FromArgb(Math.Min(255, Math.Max(0, (int)(v * 256))), basecolor);
		}

		public static Func<T, double> Normalize<T>(T xmin, T xmax)
		{
			var dxmin = Convert.ToDouble(xmin);
			var dxmax = Convert.ToDouble(xmax);
			var ddiff = dxmax - dxmin;

			return (x) => (Convert.ToDouble(x) - dxmin) / ddiff;
		}

		public static void Plot<T>(string filename, NumCIL.Generic.NdArray<T> array, Func<int, int, T, Color> encode)
		{
			Plot(filename, array, encode, Color.White);
		}

		public static void Plot<T>(string filename, NumCIL.Generic.NdArray<T> array, Func<int, int, T, Color> encode, Color backgroundColor)
		{
			var shape = array.Shape;
			if (shape.Dimensions.Length != 2)
				throw new Exception("Must be 2D array");
				
			var w = (int)shape.Dimensions[0].Length;
			var h = (int)shape.Dimensions[1].Length;

			using (var bmp = new Bitmap(w, h))
			{
				using (var gc = System.Drawing.Graphics.FromImage(bmp))
					gc.Clear(backgroundColor);

				using(var lb = new LockBitmap(bmp))
					for(var y = 0; y < h; y++)
						for(var x = 0; x < w; x++)
							lb.SetPixel(x, y, encode(x, y, array.Value[y, x]));
	
				bmp.Save(filename, System.Drawing.Imaging.ImageFormat.Png);
			}
		}
	}
}

