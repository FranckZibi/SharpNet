using System;

namespace SharpNet.Pictures
{
	public abstract class AbstractAccumulator<T>
	{
		public abstract int Count { get; }
		public abstract T Average { get; }
		public abstract void Add(T t, int count0);
		public void Add(T t)
		{
			Add(t, 1);
		}
		public abstract void Clear();
	}

	public class DoubleAccumulator : AbstractAccumulator<double>
	{
		#region private fields
		private double sum;
		private double sumSquare;
		private int count;
        #endregion

		public override int Count { get { return count; } }
		public override double Average
		{
			get { return (Count == 0) ? 0 : (sum / Count); }
		}
        public override void Add(double t, int count0)
        {
            sum += t * count0;
            sumSquare += (t * t) * count0;
            count += count0;
        }
		public override void Clear()
		{
			sum = 0;
			sumSquare = 0;
			count = 0;
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