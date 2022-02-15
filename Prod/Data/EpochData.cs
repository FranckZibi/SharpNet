using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.Models;

namespace SharpNet.Data
{
    public class EpochData
    {
        #region properties
        public double LearningRateMultiplicativeFactorFromReduceLrOnPlateau { get; }
        public readonly IDictionary<MetricEnum, double> TrainingMetrics = new Dictionary<MetricEnum, double>();
        public readonly IDictionary<MetricEnum, double> ValidationMetrics = new Dictionary<MetricEnum, double>();
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
        public EpochData(int index, double learningRateAtEpochStart, double learningRateMultiplicativeFactorFromReduceLROnPlateau, double secondsForEpoch, IDictionary<MetricEnum, double> trainingMetrics, IDictionary<MetricEnum, double> validationMetrics)
        {
            _index = index;
            LearningRateAtEpochStart = learningRateAtEpochStart;
            LearningRateMultiplicativeFactorFromReduceLrOnPlateau = learningRateMultiplicativeFactorFromReduceLROnPlateau;
            ValidationMetrics.Clear();
            SecondsForEpoch = secondsForEpoch;
            foreach (var (metric, metricValue) in validationMetrics ?? new Dictionary<MetricEnum, double>())
            {
                ValidationMetrics[metric] = metricValue;
            }
            TrainingMetrics.Clear();
            foreach (var (metric, metricValue) in trainingMetrics ?? new Dictionary<MetricEnum, double>())
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

            foreach (var metric in Enum.GetValues(typeof(MetricEnum)).Cast<MetricEnum>())
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
                   +" - "+ IModel.MetricsToString(TrainingMetrics, "Training")
                   +" - "+ IModel.MetricsToString(ValidationMetrics, "Validation");
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
        public double TrainingLoss => GetWithDefaultValue(TrainingMetrics, MetricEnum.Loss, double.NaN);
        public double ValidationLoss => GetWithDefaultValue(ValidationMetrics, MetricEnum.Loss, double.NaN);
        // ReSharper disable once UnusedMember.Global
        public double TrainingAccuracy => GetWithDefaultValue(TrainingMetrics, MetricEnum.Accuracy, double.NaN);
        public double ValidationAccuracy => GetWithDefaultValue(ValidationMetrics, MetricEnum.Accuracy, double.NaN);
        // ReSharper disable once UnusedMember.Global
        public double TrainingMae => GetWithDefaultValue(TrainingMetrics, MetricEnum.Mae, double.NaN);
        // ReSharper disable once UnusedMember.Global
        public double ValidationMae => GetWithDefaultValue(ValidationMetrics, MetricEnum.Mae, double.NaN);

        private static double GetWithDefaultValue(IDictionary<MetricEnum, double> allAvailableMetrics, MetricEnum metricEnum, double defaultValue)
        {
            return allAvailableMetrics.ContainsKey(metricEnum)
                ? allAvailableMetrics[metricEnum]
                : defaultValue;
        }
        private static IDictionary<MetricEnum, double> ToDictionary(double loss, double accuracy)
        {
            return new Dictionary<MetricEnum, double>
            {
                [MetricEnum.Loss] = loss,
                [MetricEnum.Accuracy] = accuracy
            };
        }
    }
}
