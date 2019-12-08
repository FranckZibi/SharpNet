namespace SharpNet.DataAugmentation
{
    public struct HSBColor
    {
        public float H { get; set; }
        public float S { get; set; }
        public float B { get; set; }

        public HSBColor(float h, float s, float b)
        {
            H = h;
            S = s;
            B = b;
        }

        //public static RGBColor HSB2RGB(double hueInDegrees, double s, double b)
        //{
        //    int h_i = ((int)(hueInDegrees / 60)) % 6;
        //    double f = hueInDegrees / 60 - ((int)(hueInDegrees / 60));

        //    double p = b * (1 - s);
        //    double q = b * (1 - f * s);
        //    double t = b * (1 - (1 - f) * s);

        //    switch (h_i)
        //    {
        //        case 0: return RGBColor.ToRGB(b, t, p);
        //        case 1: return RGBColor.ToRGB(q, b, p);
        //        case 2: return RGBColor.ToRGB(p, b, t);
        //        case 3: return RGBColor.ToRGB(p, q, b);
        //        case 4: return RGBColor.ToRGB(t, p, b);
        //        //case 5: 
        //        default: return RGBColor.ToRGB(b, p, q);
        //    }
        //}

        public static HSBColor RGB2HSB(byte r, byte g, byte b)
        {
            double red = r / 255.0;
            double green = g / 255.0;
            double blue = b / 255.0;

            return new HSBColor(
                (float)RGB2Hue(red, green, blue),
                (float)RGB2Saturation(red, green, blue),
                (float)RGB2Brightness(red, green, blue));
        }

        public static double RGB2Hue(double red, double green, double blue)
        {
            double max = Utils.Max(red, green, blue);
            double min = Utils.Min(red, green, blue);
            if (max == min)
            {
                return 0;
            }

            if (max == red)
            {
                return (60.0 * (green - blue) / (max - min) + 360.0) % 360.0;
            }

            if (max == green)
            {
                return (60.0 * (blue - red) / (max - min) + 120.0) % 360.0;
            }

            return (60.0 * (red - green) / (max - min) + 240.0) % 360.0;
        }
        public static double RGB2Saturation(double red, double green, double blue)
        {
            double max = Utils.Max(red, green, blue);
            if (max == 0)
            {
                return 0;
            }

            double min = Utils.Min(red, green, blue);
            return 1.0 - (min / max);
        }


        public static double RGB2Brightness(double red, double green, double blue)
        {
            return Utils.Max(red, green, blue);
        }
    }
}