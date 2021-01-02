using System;

namespace SharpNet.Pictures
{
    public sealed class ColorAccumulator
    {
        #region private fields
        private readonly DoubleAccumulator L = new DoubleAccumulator();
        private readonly DoubleAccumulator A = new DoubleAccumulator();
        private readonly DoubleAccumulator B = new DoubleAccumulator();
        #endregion

        public override int GetHashCode()
        {
            return Count == 0 ? 0 : Average.GetHashCode();
        }
        public override string ToString()
        {
            return "E(Color)=" + Average + " ; Volatility="+Math.Round(Volatility,1)+" ; "  + Count + "pixels;";
        }

        private int Count => L.Count;

        public void Add(RGBColor color, int count = 1)
        {
            if (color == null)
            {
                return;
            }

            L.Add(color.Lab.L, count);
            A.Add(color.Lab.A, count);
            B.Add(color.Lab.B, count);
        }
        public RGBColor Average
        {
            get
            {
                return Count == 0 ? null : LabColor.Lab2RGB(L.Average, A.Average, B.Average);
            }
        }

        private double Volatility
        {
            get { return Math.Sqrt(Math.Pow(L.Volatility, 2) + Math.Pow(A.Volatility, 2) + Math.Pow(B.Volatility, 2)); }
        }
    }
}
