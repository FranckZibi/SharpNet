using System;
// ReSharper disable CompareOfFloatsByEqualityOperator

namespace SharpNet.Pictures
{
    public class LabColor
    {
        //Observer= 2°, Illuminant= D65
        private const double XYZ_REF_WHITE_X = 95.047;
        private const double XYZ_REF_WHITE_Y = 100;
        private const double XYZ_REF_WHITE_Z = 108.883;

        private LabColor(double l, double a, double b)
        {
            L = l;
            A = a;
            B = b;
        }
        public double L { get; }
        public double A { get; }
        public double B { get; }

        public double Distance(LabColor lab)
        {
            double deltaL = lab.L - L;
            double deltaA = lab.A - A;
            double deltaB = lab.B - B;
            return Math.Sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB) / 100;

          
        }
        public static RGBColor Lab2RGB(double L, double A, double B)
        {
            const double eps = 216.0 / 24389.0;
            const double k = 24389.0 / 27.0;

            double f_y = (L + 16.0) / 116.0;
            double f_z = f_y - (B / 200.0);
            double f_x = (A / 500.0) + f_y;
            double x_r = Math.Pow(f_x, 3.0);
            if (x_r <= eps)
            {
                x_r = (116 * f_x - 16) / k;
            }

            double y_r = (L > (k * eps)) ? Math.Pow((L + 16) / 116, 3.0) : L / k;
            double z_r = Math.Pow(f_z, 3.0);
            if (z_r <= eps)
            {
                z_r = (116 * f_z - 16) / k;
            }

            double X = x_r * XYZ_REF_WHITE_X;
            double Y = y_r * XYZ_REF_WHITE_Y;
            double Z = z_r * XYZ_REF_WHITE_Z;
            return new XYZColor(X, Y, Z).XYZ2RGB();
        }
        public static LabColor RGB2Lab(byte r, byte g, byte b)
        {
            var result = new double[3];
            var xyz = XYZColor.RGB2XYZ(r, g, b);
            double x_prime = xyz.X / XYZ_REF_WHITE_X;
            double y_prime = xyz.Y / XYZ_REF_WHITE_Y;
            double z_prime = xyz.Z / XYZ_REF_WHITE_Z;
            result[0] = 116.0 * LabFunction(y_prime) - 16.0;
            result[1] = 500.0 * (LabFunction(x_prime) - LabFunction(y_prime));
            result[2] = 200.0 * (LabFunction(y_prime) - LabFunction(z_prime));
            return new LabColor(result[0], result[1], result[2]);
        }
        public override string ToString()
        {
            return Math.Round(L, 2) + ";" + Math.Round(A, 2) + ";" + Math.Round(B, 2);
        }
        public override int GetHashCode()
        {
            return L.GetHashCode() + A.GetHashCode() + B.GetHashCode();
        }
        public override bool Equals(object obj)
        {
            if (!(obj is LabColor))
            {
                return false;
            }

            var other = (LabColor)obj;
            return (L == other.L) && (A == other.A) && (B == other.B);
        }
        private static double LabFunction(double t)
        {
            const double eps = 216.0 / 24389.0;
            const double k = 24389.0 / 27.0;

            if (t > eps)
            {
                return Math.Pow(t, 1 / 3.0);
            }

            return ((k * t + 16.0) / 116.0);
        }
        private double C { get { return Math.Sqrt(A * A + B * B); } }

    }
}