using System;
using System.Collections.Generic;

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

        public long Count { get; private set; }
        public double Average => (Count == 0) ? 0 : (sum / Count);
        public double Min => min;
        public double Max => max;

        public void Add(double t, int count = 1)
        {
            min = (Count == 0) ? t : Math.Min(t, min);
            max = (Count == 0) ? t : Math.Max(t, max);
            sum += t * count;
            sumSquare += (t * t) * count;
            Count += count;
        }
        public void Add(IEnumerable<float> list)
        {
            foreach (var e in list)
            {
                Add(e);
            }
        }
        public void Add(Span<float> span)
        {
            foreach (var e in span)
            {
                Add(e);
            }
        }
        // ReSharper disable once UnusedMember.Global
        public void Add(IEnumerable<double> list)
        {
            foreach (var e in list)
            {
                Add(e);
            }
        }
        public void Add(DoubleAccumulator b)
        {
            min = (Count == 0) ? b.min : Math.Min(b.min, min);
            max = (Count == 0) ? b.max : Math.Max(b.max, max);
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
		public double Volatility => Math.Sqrt(Variance);


        /// <summary>
        /// Coefficient Of Variation, see https://en.wikipedia.org/wiki/Coefficient_of_variation
        /// Volatility / Average 
        /// </summary>
        // ReSharper disable once UnusedMember.Global
        public double CoefficientOfVariation => (Math.Abs(Average) < 1e-6) ? 0 : (Volatility / Average);

        public double Variance
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
            string result = "E(x)=" + Math.Round(Average, roundDigits) + "; Vol(X)=" + Math.Round(Volatility, roundDigits) + "; Count=" + Count + "; Min=" + min + "; Max=" + max;
            return result;
        }
    }
}
