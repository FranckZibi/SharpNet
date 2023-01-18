using System;
using System.Collections.Generic;

namespace SharpNet.Pictures;

public class RGBColor : IEquatable<RGBColor>
{
    #region private fields
    private readonly byte red;
    private readonly byte green;
    private readonly byte blue;
    public int Index24Bits { get; }
    private LabColor _lazyLab;
    #endregion

    public static byte ToRGBColor(double color_255)
    {
        return (byte)Utils.NearestInt(255 * color_255);
    }


    public static readonly RGBColor Black = new RGBColor(0, 0, 0);
    public static readonly RGBColor White = new RGBColor(255, 255, 255);
    public static readonly RGBColor RedRef = new RGBColor(255, 0, 0);
    public static readonly RGBColor OrangeRef = new RGBColor(255, 165, 0);
    public static readonly RGBColor GreenRef = new RGBColor(0, 255, 0);
    public static readonly RGBColor PinkRef = new RGBColor(255, 102, 204);
    public static readonly RGBColor GreyRef = new RGBColor(161, 169, 164);

    public static readonly RGBColor DarkBlue = new RGBColor(0, 0, 139);
    public static readonly RGBColor ClearBlue = new RGBColor(135, 206, 235);
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

    private static double ColorDistance(RGBColor a, RGBColor b)
    {
        return a.Lab.Distance(b.Lab);
    }
    public double ColorDistance(RGBColor b)
    {
        return ColorDistance(this,b);
    }

    public static double HueDistanceInDegrees(double hue1InDegrees, double hue2InDegrees)
    {
        double hueDistanceInDegrees = Math.Abs(hue1InDegrees - hue2InDegrees);
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

    public static void SortByLabBrightestFirst(List<RGBColor> colors)
    {
        colors.Sort((x, y) => (int)(1000000 * (y.Lab.L - x.Lab.L)));
    }

    public double MinDistanceTo(IEnumerable<RGBColor> colors, Func<RGBColor, RGBColor, double> ColorDistance)
    {
        double result = double.MaxValue;
        foreach (var c in colors)
        {
            result = Math.Min(ColorDistance(this, c), result);
        }

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
        return other != null && Index24Bits == other.Index24Bits;
    }
    public double MaxDistanceTo(IEnumerable<RGBColor> colors, Func<RGBColor, RGBColor, double> ColorDistance)
    {
        double result = 0;
        foreach (var c in colors)
        {
            result = Math.Max(ColorDistance(this, c), result);
        }

        return result;
    }


    public static RGBColor ToRGB(double r_255, double g_255, double b_255)
    {
        return new RGBColor(ToRGBColor(r_255), ToRGBColor(g_255), ToRGBColor(b_255));
    }
    public static double LabColorDistance(RGBColor a, RGBColor b)
    {
        return a.Lab.Distance(b.Lab);
    }
    public static double BlackLabColorDistance(RGBColor a, RGBColor b)
    {
        var res = a.Lab.Distance(b.Lab);

        const double mult = 4.0;
        if (b.Red == 0)
        {
            res /= mult;
        }
        if (b.Green == 0)
        {
            res /= mult;
        }
        if (b.Blue == 0)
        {
            res /= mult;
        }

        //var min = Math.Min(b.Red, Math.Min(b.Green, b.Blue));
        //if (min < 10)
        //{
        //    res /= mult;
        //}


        if (b.Red != 0 && b.Green != 0 && b.Blue != 0)
        {
            res *= mult;
        }

        int sum = b.Red + b.Green + b.Blue;
        if (sum > 20.0)
        {
            res *= sum/20.0;
        }
        
        //if (sum < 10.0)
        //{
        //    res *= (10-sum) / 10.0;
        //}


        //if (b.Red > 5 && b.Green > 5 && b.Blue > 5)
        //{
        //    res *= mult;
        //}

        var max = Math.Max(b.Red, Math.Max(b.Green, b.Blue));
        if (max >= 30)
        {
            return 1.0;
        }

        //if (max <= 10)
        //{
        //    return 0.0;
        //}

        return res;
    }


    public static double YellowLabColorDistance(RGBColor a, RGBColor b)
    {
        var aLab = a.Lab;
        var bLab = b.Lab;
        //var res0 = aLab.Distance(bLab);


        var res = aLab.Distance(bLab);
        return res;

        if (b.Blue > 75)
        {
            return 1.0;
        }
            

        var errorR = 255 - b.Red;
        var errorG = 255 - b.Green;
        var errorB = b.Blue;

        int sum = errorR + errorG + errorB;
        if (sum > 50.0)
        {
            //res *= sum / 20.0;
            return 1.0;
        }

        const double mult = 4.0;
        if (errorR == 0)
        {
            res /= mult;
        }
        if (errorG == 0)
        {
            res /= mult;
        }
        if (errorB == 0)
        {
            res /= mult;
        }


        



        if (errorR != 0 && errorG != 0 && errorB != 0)
        {
            res *= mult;
        }




        



        var max = Math.Max(errorR, Math.Max(errorG, errorB));
        if (max >= 40)
        {
            return 1.0;
        }

        return res;


        //if (max <= 10)
        //{
        //    return 0.0;
        //}

        return res;
    }

    public static double WhiteLabColorDistance(RGBColor a, RGBColor b)
    {
        var aLab = a.Lab;
        var bLab = b.Lab;
        //var res0 = aLab.Distance(bLab);

        //if (Math.Abs(bLab.B) > 5)
        //{
        //    return 1.0;
        //}

        //var res1 = a.HSB2.DistanceISS(b.HSB2);
        //return res0;

        var res = aLab.Distance(bLab); ;

        var errorR = 255 - b.Red;
        var errorG = 255 - b.Green;
        var errorB = 255 - b.Blue;
        

        const double mult = 4.0;
        if (errorR == 0)
        {
            res /= mult;
        }
        if (errorG == 0)
        {
            res /= mult;
        }
        if (errorB == 0)
        {
            res /= mult;
        }


      

        if (errorR != 0 && errorG != 0 && errorB != 0)
        {
            res *= mult;
        }



        int sum = errorR + errorG + errorB;
        if (sum > 20.0)
        {
            res *= sum / 20.0;
        }


        //if (sum < 10.0)
        //{
        //    res *= (10-sum) / 10.0;
        //}


        //if (b.Red > 5 && b.Green > 5 && b.Blue > 5)
        //{
        //    res *= mult;
        //}

        var max = Math.Max(errorR, Math.Max(errorG, errorB));
        if (max >= 35)
        {
            return 1.0;
        }

        return res;


        //if (max <= 10)
        //{
        //    return 0.0;
        //}

        return res;
    }
}