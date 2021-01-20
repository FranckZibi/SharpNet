namespace SharpNet.MathTools
{
    public sealed class DoubleAccumulator
	{
		#region private fields
		private double sum;
		private double sumSquare;
        private double min;
        private double max;
        #endregion

        public int Count { get; private set; }
        public double Average => (Count == 0) ? 0 : (sum / Count);

		public void Add(double t, int count)
        {
            min = (Count == 0) ? t : System.Math.Min(t, min);
            max = (Count == 0) ? t : System.Math.Max(t, max);
            sum += t * count;
            sumSquare += (t * t) * count;
            Count += count;

        }
        private void Add(DoubleAccumulator b)
        {
            min = (Count == 0) ? b.min : System.Math.Min(b.min, min);
            max = (Count == 0) ? b.max : System.Math.Max(b.max, max);
            sum += b.sum;
            sumSquare += b.sumSquare;
            Count += b.Count;
        }
        public static DoubleAccumulator Sum(params DoubleAccumulator[] accumulators)
        {
            var res = new DoubleAccumulator();
            foreach (var a in accumulators)
            {
                res.Add(a);
            }

            return res;
        }
		public double Volatility => System.Math.Sqrt(Variance);
        private double Variance
		{
			get
			{
                if (Count <= 0)
                {
                    return 0;
                }

                return System.Math.Abs(sumSquare - Count * Average * Average) / Count;
			}
		}
		public override string ToString()
		{
			return ToString(6);
		}
        private string ToString(int roundDigits)
        {
            string result = "E(x)=" + System.Math.Round(Average, roundDigits) + "; Vol(X)=" + System.Math.Round(Volatility, roundDigits) + "; Count=" + Count + "; Min=" + min + "; Max=" + max;
            return result;
        }
    }
}
