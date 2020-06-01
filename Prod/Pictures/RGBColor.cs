using System;
using System.Collections.Generic;

namespace SharpNet.Pictures
{
    public class RGBColor : IEquatable<RGBColor>
    {
        #region private fields
        private readonly byte red;
        private readonly byte green;
        private readonly byte blue;
        private int Index24Bits { get; }
        private LabColor _lazyLab;
        #endregion

        public static RGBColor ToRGB(double r_255, double g_255, double b_255)
        {
            return new RGBColor(ToRGBColor(r_255), ToRGBColor(g_255), ToRGBColor(b_255));
        }

        public static byte ToRGBColor(double color_255)
        {
            return (byte)Utils.NearestInt(255 * color_255);
        }




        public static readonly RGBColor Black = new RGBColor(0, 0, 0);
		public static readonly RGBColor White = new RGBColor(255, 255, 255);


		public RGBColor(byte red, byte green, byte blue)
		{
			this.red = red;
			this.green = green;
			this.blue = blue;
            Index24Bits = red + (green << 8) + (blue << 16);
		}
        public byte Red => red;
        public byte Green => green;
        public byte Blue => blue;

        // ReSharper disable UnusedMember.Global
		public double DistanceToWhite { get { return ColorDistance(this, White); } }
        public double DistanceToBlack { get { return ColorDistance(this, Black); } }


        public LabColor Lab
        {
            get
            {
                if (_lazyLab == null)
                {
                    _lazyLab = LabColor.RGB2Lab(red, green, blue);
                }

                return _lazyLab;
            }
        }
    
        public List<double> Distance(IEnumerable<RGBColor> colors)
        {
            var result = new List<double>();
            foreach (var c in colors)
            {
                result.Add(ColorDistance(this, c));
            }

            return result;
        }

        public static double ColorDistance(RGBColor a, RGBColor b)
        {
            return a.Lab.Distance(b.Lab);
        }

        public double ColorDistance(RGBColor b)
        {
            return ColorDistance(this,b);
        }



        public static double HueDistanceInDegrees(double hue1InDegrees, double hue2InDegrres)
        {
            double hueDistanceInDegrees = Math.Abs(hue1InDegrees - hue2InDegrres);
            if (hueDistanceInDegrees >= 180)
            {
                hueDistanceInDegrees = 360 - hueDistanceInDegrees;
            }

            return hueDistanceInDegrees;
        }
       


        public static RGBColor Average(List<RGBColor> colors)
        {
            if (colors.Count == 1)
            {
                return colors[0];
            }

            var acc = new ColorAccumulator();
            foreach(var c in colors)
            {
                acc.Add(c);
            }

            return acc.Average;
        }

        public static RGBColor PonderedAverage(List<KeyValuePair<RGBColor, int>> colors)
        {
            if (colors.Count == 1)
            {
                return colors[0].Key;
            }

            var acc = new ColorAccumulator();
            foreach (var c in colors)
            {
                acc.Add(c.Key, c.Value);
            }

            return acc.Average;
        }
        public override string ToString()
        {
			string result = ("(" + Red + "," + Green + "," + Blue + ")" + "(" + Lab + ")");
            return result;
        }

        public override int GetHashCode()
        {
            return Index24Bits;
        }

        public override bool Equals(object obj)
        {
            if (!(obj is RGBColor))
            {
                return false;
            }

            return Equals((RGBColor)obj);
        }
        public bool Equals(RGBColor other)
        {
            return Index24Bits == other.Index24Bits;
        }
    }
}