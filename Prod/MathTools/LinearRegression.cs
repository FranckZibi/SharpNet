using System;

namespace SharpNet.MathTools
{
    /// <summary>
    /// Compute a linear regression 
    ///     y = Slope * x + Intercept
    /// with:
    ///     x: the independent variable
    ///     y: the dependent variable
    /// </summary>
    public sealed class LinearRegression
    {
        #region private fields
        private double xy_sum;
        private double x_sum;
        private double xx_sum;
        private double y_sum;
        private double yy_sum;
        private int count;
        #endregion


        /// <summary>
        /// add the observation (x, y) that will be used for the linear regression
        /// </summary>
        /// <param name="x">independent variable</param>
        /// <param name="y">dependent variable</param>
        public void Add(double x, double y)
        {
            if (double.IsNaN(y))
            {
                return;
            }
            ++count;
            xy_sum += x * y;
            x_sum += x;
            xx_sum += x * x;
            y_sum += y;
            yy_sum += y*y;
        }


        /// <summary>
        /// the slope
        /// </summary>
        public double Slope
        {
            get
            {
                if (count == 0)
                {
                    return 0;
                }
                return (count * xy_sum - x_sum * y_sum) / (count * xx_sum - x_sum * x_sum);
            }
        }


        /// <summary>
        /// the y-intercept
        /// </summary>
        public double Intercept
        {
            get
            {
                if (count == 0)
                {
                    return 0;
                }
                return (y_sum - Slope * x_sum) / count;
            }
        }



        public double Y_Average => (count == 0) ? 0 : (y_sum / count);
        public double Y_Volatility => System.Math.Sqrt(Y_Variance);
        public double Y_Variance
        {
            get
            {
                if (count <= 0)
                {
                    return 0;
                }
                return Math.Abs(yy_sum - count * Y_Average * Y_Average) / count;
            }
        }


        /// <summary>
        /// estimate value of the dependent variable 'y' given the independent variable 'x'
        /// </summary>
        /// <param name="x">independent variable</param>
        public double Estimation(double x)
        {
            return Slope * x + Intercept;
        }

        /// <summary>
        /// Coefficient of determination: proportion of the variance (in the dependent variable 'y') that is predictable from the independent variable 'y'
        /// Cf. https://en.wikipedia.org/wiki/Coefficient_of_determination
        /// </summary>
        public double RSquared => PearsonCorrelationCoefficient* PearsonCorrelationCoefficient;

        /// <summary>
        /// Pearson correlation coefficient (PCC) = Pearson's r = Pearson product-moment correlation coefficient (PPMCC) = bi-variate correlation
        /// https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        /// </summary>
        public double PearsonCorrelationCoefficient
        {
            get
            {
                if (count == 0)
                {
                    return 0;
                }
                var upper = count * xy_sum - x_sum * y_sum;
                var lower = Math.Sqrt( (count * xx_sum - x_sum * x_sum) * (count * yy_sum - y_sum * y_sum) );
                return upper / lower;
            }
        }

        public override string ToString()
        {
            return ToString(6);
        }
        private string ToString(int roundDigits)
        {
            string result = "Y = "+Math.Round(Slope, roundDigits) + " *X  +  " + Math.Round(Intercept, roundDigits) + " ; R^2 = " + Math.Round(RSquared, roundDigits)+ "; Count=" + count;
            return result;
        }
    }
}