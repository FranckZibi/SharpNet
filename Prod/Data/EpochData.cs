using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.Networks;

namespace SharpNet.Data
{
    public class EpochData
    {
        #region properties
        public double LearningRateMultiplicativeFactorFromReduceLrOnPlateau { get; }
        public readonly IDictionary<NetworkConfig.Metric, double> TrainingMetrics = new Dictionary<NetworkConfig.Metric, double>();
        public readonly IDictionary<NetworkConfig.Metric, double> ValidationMetrics = new Dictionary<NetworkConfig.Metric, double>();
        private readonly int _index;
        private double LearningRateAtEpochStart { get; }
        private double SecondsForEpoch { get; }
        #endregion

        public EpochData(int index, double learningRateAtEpochStart, double learningRateMultiplicativeFactorFromReduceLROnPlateau, 
            double trainingLoss, double trainingAccuracy, double validationLoss, double validationAccuracy, 
            double secondsForEpoch)
            : this(index, learningRateAtEpochStart, learningRateMultiplicativeFactorFromReduceLROnPlateau, secondsForEpoch, 
                ToDictionary(trainingLoss, trainingAccuracy), 
                ToDictionary(validationLoss, validationAccuracy)) 
        {
        }
        public EpochData(int index, double learningRateAtEpochStart, double learningRateMultiplicativeFactorFromReduceLROnPlateau, double secondsForEpoch, IDictionary<NetworkConfig.Metric, double> trainingMetrics, IDictionary<NetworkConfig.Metric, double> validationMetrics)
        {
            _index = index;
            LearningRateAtEpochStart = learningRateAtEpochStart;
            LearningRateMultiplicativeFactorFromReduceLrOnPlateau = learningRateMultiplicativeFactorFromReduceLROnPlateau;
            ValidationMetrics.Clear();
            SecondsForEpoch = secondsForEpoch;
            foreach (var (metric, metricValue) in validationMetrics ?? new Dictionary<NetworkConfig.Metric, double>())
            {
                ValidationMetrics[metric] = metricValue;
            }
            TrainingMetrics.Clear();
            foreach (var (metric, metricValue) in trainingMetrics ?? new Dictionary<NetworkConfig.Metric, double>())
            {
                TrainingMetrics[metric] = metricValue;
            }
        }

        #region serialization
        public string Serialize()
        {
            var serializer = new Serializer()
                .Add(nameof(_index), _index)
                .Add(nameof(LearningRateAtEpochStart), LearningRateAtEpochStart)
                .Add(nameof(LearningRateMultiplicativeFactorFromReduceLrOnPlateau), LearningRateMultiplicativeFactorFromReduceLrOnPlateau)
                .Add(nameof(SecondsForEpoch), SecondsForEpoch);
            foreach (var (metric, metricValue) in ValidationMetrics)
            {
                serializer.Add("Validation" + metric, metricValue);
            }
            foreach (var (metric, metricValue) in TrainingMetrics)
            {
                serializer.Add("Training" + metric, metricValue);
            }
            return serializer.ToString();
        }
        public EpochData(IDictionary<string, object> serialized)
        {
            _index = (int)serialized[nameof(_index)];
            LearningRateAtEpochStart = (double)serialized[nameof(LearningRateAtEpochStart)];
            LearningRateMultiplicativeFactorFromReduceLrOnPlateau = (double)serialized[nameof(LearningRateMultiplicativeFactorFromReduceLrOnPlateau)];

            foreach (var metric in Enum.GetValues(typeof(NetworkConfig.Metric)).Cast<NetworkConfig.Metric>())
            {
                if (serialized.TryGetValue("Validation" + metric, out var validationMetric))
                {
                    ValidationMetrics[metric] = (double)validationMetric;
                }
                if (serialized.TryGetValue("Training" + metric, out var trainingMetric))
                {
                    TrainingMetrics[metric] = (double)trainingMetric;
                }
            }
            SecondsForEpoch = (double)serialized[nameof(SecondsForEpoch)];
        }
        #endregion

        public override string ToString()
        {
            return "Epoch "+_index+" : learningRate:"+LearningRateAtEpochStart
                   +" - "+ Network.MetricsToString(TrainingMetrics, "Training")
                   +" - "+ Network.MetricsToString(ValidationMetrics, "Validation");
        }
        public bool Equals(EpochData other, double epsilon)
        {
            return _index == other._index
                   && Math.Abs(LearningRateAtEpochStart - other.LearningRateAtEpochStart) <= epsilon
                   && Math.Abs(LearningRateMultiplicativeFactorFromReduceLrOnPlateau - other.LearningRateMultiplicativeFactorFromReduceLrOnPlateau) <= epsilon
                   && Math.Abs(SecondsForEpoch - other.SecondsForEpoch) <= epsilon
                //TODO : compare also ValidationMetrics & TrainingMetrics
                ;
        }
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj))
            {
                return false;
            }
            if (ReferenceEquals(this, obj))
            {
                return true;
            }
            if (obj.GetType() != GetType())
            {
                return false;
            }
            return _index == ((EpochData)obj)._index;
        }
        public override int GetHashCode()
        {
            return _index;
        }
        public double TrainingLoss => GetWithDefaultValue(TrainingMetrics, NetworkConfig.Metric.Loss, double.NaN);
        public double ValidationLoss => GetWithDefaultValue(ValidationMetrics, NetworkConfig.Metric.Loss, double.NaN);
        public double TrainingAccuracy => GetWithDefaultValue(TrainingMetrics, NetworkConfig.Metric.Accuracy, double.NaN);
        public double ValidationAccuracy => GetWithDefaultValue(ValidationMetrics, NetworkConfig.Metric.Accuracy, double.NaN);
        public double TrainingMae => GetWithDefaultValue(TrainingMetrics, NetworkConfig.Metric.Mae, double.NaN);
        public double ValidationMae => GetWithDefaultValue(ValidationMetrics, NetworkConfig.Metric.Mae, double.NaN);

        private static double GetWithDefaultValue(IDictionary<NetworkConfig.Metric, double> allAvailableMetrics, NetworkConfig.Metric metric, double defaultValue)
        {
            return allAvailableMetrics.ContainsKey(metric)
                ? allAvailableMetrics[metric]
                : defaultValue;
        }
        private static IDictionary<NetworkConfig.Metric, double> ToDictionary(double loss, double accuracy)
        {
            return new Dictionary<NetworkConfig.Metric, double>
            {
                [NetworkConfig.Metric.Loss] = loss,
                [NetworkConfig.Metric.Accuracy] = accuracy
            };
        }
    }
}
