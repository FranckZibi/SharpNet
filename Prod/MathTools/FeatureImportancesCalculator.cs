using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace SharpNet.MathTools
{
    public class FeatureImportancesCalculator
    {
        /// <summary>
        /// Compute extra features based on current features
        /// Ex: exp(feature) , log(feature), vol(feature), var(feature), etc.
        /// </summary>
        private readonly bool ComputeExtraFeature;
        private readonly List<string> FeatureNames = new List<string>();
        private readonly List<List<double>> FeatureValues = new List<List<double>>();
        private readonly List<double> Labels = new List<double>();
        private readonly List<double> CurrentFeatures = new List<double>();
        private double SumLabels;
        private double SumSquareLabels;
        private readonly List<double> SumFeatures = new List<double>();
        private readonly List<double> SumSquareFeatures = new List<double>();
        private readonly List<double> FeaturesMinimum = new List<double>();
        private readonly List<double> FeaturesMaximum = new List<double>();
        private int Count => FeatureValues.Count;
        private int FeatureCount => FeatureNames.Count;

        /// <param name="computeExtraFeature">true if we should compute extra features based on current feature</param>
        public FeatureImportancesCalculator(bool computeExtraFeature)
        {
            ComputeExtraFeature = computeExtraFeature;
        }


        public void AddFeature(double data, string name)
        {
            AddSingleFeature(data, name);
            if (ComputeExtraFeature)
            {
                AddSingleFeature((float) Math.Exp(data), "exp("+name+")");
                AddSingleFeature((float) Math.Log(Math.Abs(data) + 1e-5), "log("+name+")");
            }
        }

        public static string VectorFeatureName(string vectorName, int indexInVector)
        {
            return vectorName + "[" + indexInVector.ToString("D2") + "]";
        }

        /// <summary>
        /// Load a CSV file with feature statistics
        /// </summary>
        /// <param name="csvFilePath">
        /// CSV file path. Expected header for the file:
        ///     FeatureName;Minimum;Maximum;Mean;Volatility;CorrelationWithLabel;FeatureImportance
        /// </param>
        /// <returns>
        /// For each feature name, a Tuple with:
        ///     Item1:  feature minimum
        ///     Item2:  feature maximum
        ///     Item3:  feature mean
        ///     Item4:  feature volatility
        ///     Item5:  feature correlation with label
        ///     Item6:  feature importances
        /// </returns>
        public static Dictionary<string, Tuple<double, double, double, double, double, double>> LoadFromFile(string csvFilePath)
        {
            var featureNameToStatistics = new Dictionary<string, Tuple<double, double, double, double, double, double>>();
            foreach (var line in System.IO.File.ReadAllLines(csvFilePath).Skip(2))
            {
                var lineContent = line.Split(';', ',');
                var featureName = lineContent[0];
                var values = lineContent.Skip(1).Select(double.Parse).ToArray();
                featureNameToStatistics[featureName] = Tuple.Create(values[0], values[1], values[2], values[3], values[4], values[5]);
            }
            return featureNameToStatistics;
        }

        public void AddFeature(float[] data, string name)
        {
            for (int i = 0; i < data.Length; ++i)
            {
                AddFeature(data[i], VectorFeatureName(name, i));
            }
            if (ComputeExtraFeature)
            {
                var acc = new DoubleAccumulator();
                acc.Add(data);
                AddSingleFeature((float) acc.Volatility, "vol("+name+")");
                AddSingleFeature((float) (acc.Variance), "var("+name+")");
                AddSingleFeature((float) (acc.Average), "mean("+name+")");
                AddSingleFeature(data.Max(), "max("+name+")");
                AddSingleFeature(data.Min(), "min("+name+")");
            }
        }
        public void AddLabel(double y)
        {
            if (FeatureValues.Count != 0 && FeatureValues[0].Count != CurrentFeatures.Count)
            {
                throw new Exception("invalid features length : " + CurrentFeatures.Count);
            }
            Labels.Add(y);
            SumLabels += y;
            SumSquareLabels += y*y;
            FeatureValues.Add(CurrentFeatures.ToList());
            CurrentFeatures.Clear();
        }

        public void Write(string filePath)
        {
            var sb = new StringBuilder();
            sb.Append("Sep=;" + Environment.NewLine);
            sb.Append("FeatureName;Minimum;Maximum;Mean;Volatility;CorrelationWithLabel;FeatureImportances" + Environment.NewLine);

            var sumLabelTimeFeature = new double[FeatureCount];

            for (int i = 0; i < Count; ++i)
            {
                var features = FeatureValues[i];
                var label = Labels[i];
                for (int featureId = 0; featureId < FeatureCount; ++featureId)
                {
                    sumLabelTimeFeature[featureId] += label*features[featureId];
                }
            }

            var correlations = new List<double>();
            for (int featureId = 0; featureId < FeatureCount; ++featureId)
            {
                //We compute the correlation:  Corr( feature[featureId], label )
                double top = (Count * sumLabelTimeFeature[featureId] - SumFeatures[featureId] * SumLabels);
                double bottom = (Count * SumSquareFeatures[featureId] - SumFeatures[featureId] * SumFeatures[featureId]) * (Count * SumSquareLabels - SumLabels * SumLabels);
                correlations.Add(top / Math.Sqrt(bottom));
            }
            var totalCorrelationWeight = correlations.Select(c=> double.IsNaN(c)?0:Math.Abs(c)).Sum();
            for (int featureId = 0; featureId < FeatureCount; ++featureId)
            {
                var featureName = FeatureNames[featureId];
                var correlationWithLabel = correlations[featureId];
                var featureImportances = Math.Abs(correlationWithLabel) / totalCorrelationWeight;
                sb.Append(featureName + ";"
                                      + FeaturesMinimum[featureId].ToString(CultureInfo.InvariantCulture) + ";"
                                      + FeaturesMaximum[featureId].ToString(CultureInfo.InvariantCulture) + ";"
                                      + FeatureMean(featureId).ToString(CultureInfo.InvariantCulture) + ";"
                                      + FeatureVolatility(featureId).ToString(CultureInfo.InvariantCulture) + ";"
                                      + correlationWithLabel.ToString(CultureInfo.InvariantCulture) + ";"
                                      + featureImportances.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
            }

            sb.Append(Environment.NewLine);
            System.IO.File.WriteAllText(filePath, sb.ToString());
        }

        private double FeatureVariance(int featureId) {return SumSquareFeatures[featureId] / Count - FeatureMean(featureId) * FeatureMean(featureId);}
        private double FeatureVolatility(int featureId){return Math.Sqrt(FeatureVariance(featureId));}
        private double FeatureMean(int featureId){return SumFeatures[featureId] / Count;}
        private void AddSingleFeature(double data, string name)
        {
            int featureId = CurrentFeatures.Count;
            CurrentFeatures.Add(data);
            if (Count == 0)
            {
                FeatureNames.Add(name);
                SumFeatures.Add(0);
                SumSquareFeatures.Add(0);
                FeaturesMinimum.Add(double.MaxValue);
                FeaturesMaximum.Add(double.MinValue);
            }
            SumFeatures[featureId] += data;
            SumSquareFeatures[featureId] += data*data;
            FeaturesMinimum[featureId] = Math.Min(FeaturesMinimum[featureId], data);
            FeaturesMaximum[featureId] = Math.Max(FeaturesMaximum[featureId], data);
        }
    }
}
