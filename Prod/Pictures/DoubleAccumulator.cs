using System;

namespace SharpNet.Pictures
{
	public sealed class DoubleAccumulator
	{
		#region private fields
		private double sum;
		private double sumSquare;
        #endregion

		public int Count { get; private set; }
        public double Average => (Count == 0) ? 0 : (sum / Count);

        public void Add(double t, int count)
        {
            sum += t * count;
            sumSquare += (t * t) * count;
            Count += count;
        }
		public void Clear()
		{
			sum = 0;
			sumSquare = 0;
			Count = 0;
		}
		public double Volatility => Math.Sqrt(Variance);
        private double Variance
		{
			get
			{
                if (Count <= 0)
                {
                    return 0;
                }

                return Math.Abs(sumSquare - Count * Average * Average) / Count;
			}
		}
		public override string ToString()
		{
			return ToString(6);
		}
        private string ToString(int roundDigits)
        {
            string result = "E(x)=" + Math.Round(Average, roundDigits) + "; Vol(X)=" + Math.Round(Volatility, roundDigits) + "; Count=" + Count;
            return result;
        }

    }
}
