using System;
using System.Drawing;

namespace Utilities
{
	public static class Render
	{
		public static void Plot<T>(string filename, NumCIL.Generic.NdArray<T> array, Func<int, int, T, Color> encode)
		{
			var shape = array.Shape;
			if (shape.Dimensions.Length != 2)
				throw new Exception("Must be 2D array");
				
			var w = (int)shape.Dimensions[0].Length;
			var h = (int)shape.Dimensions[1].Length;

			using (var bmp = new Bitmap(w, h))
			{
				for(var y = 0; y < h; y++)
					for(var x = 0; x < w; x++)
						bmp.SetPixel(x, y, encode(x, y, array.Value[y, x]));

				bmp.Save(filename, System.Drawing.Imaging.ImageFormat.Png);
			}
		}
	}
}

