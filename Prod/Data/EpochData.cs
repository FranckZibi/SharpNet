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
        public readonly IDictionary<EvaluationMetricEnum, double> TrainingMetrics = new Dictionary<EvaluationMetricEnum, double>();
        public readonly IDictionary<EvaluationMetricEnum, double> ValidationMetrics = new Dictionary<EvaluationMetricEnum, double>();
        private readonly int _index;
        private double LearningRateAtEpochStart { get; }
        private double SecondsForEpoch { get; }
        #endregion

        /// <summary>
        /// this constructor is needed only for tests
        /// </summary>
        /// <param name="index"></param>
        /// <param name="learningRateAtEpochStart"></param>
        /// <param name="learningRateMultiplicativeFactorFromReduceLROnPlateau"></param>
        /// <param name="trainingLoss"></param>
        /// <param name="trainingAccuracy"></param>
        /// <param name="validationLoss"></param>
        /// <param name="validationAccuracy"></param>
        /// <param name="secondsForEpoch"></param>
        /// <param name="loss"></param>
        public EpochData(int index, double learningRateAtEpochStart,
            double learningRateMultiplicativeFactorFromReduceLROnPlateau,
            double trainingLoss, double trainingAccuracy, double validationLoss, double validationAccuracy,
            double secondsForEpoch, EvaluationMetricEnum loss)
            : this(index, learningRateAtEpochStart, learningRateMultiplicativeFactorFromReduceLROnPlateau, secondsForEpoch, 
                ToDictionary(trainingLoss, trainingAccuracy, loss), 
                ToDictionary(validationLoss, validationAccuracy, loss)) 
        {
        }
        public EpochData(int index, double learningRateAtEpochStart, double learningRateMultiplicativeFactorFromReduceLROnPlateau, double secondsForEpoch, IDictionary<EvaluationMetricEnum, double> trainingMetrics, IDictionary<EvaluationMetricEnum, double> validationMetrics)
        {
            _index = index;
            LearningRateAtEpochStart = learningRateAtEpochStart;
            LearningRateMultiplicativeFactorFromReduceLrOnPlateau = learningRateMultiplicativeFactorFromReduceLROnPlateau;
            ValidationMetrics.Clear();
            SecondsForEpoch = secondsForEpoch;
            foreach (var (metric, metricValue) in validationMetrics ?? new Dictionary<EvaluationMetricEnum, double>())
            {
                ValidationMetrics[metric] = metricValue;
            }
            TrainingMetrics.Clear();
            foreach (var (metric, metricValue) in trainingMetrics ?? new Dictionary<EvaluationMetricEnum, double>())
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

            foreach (var metric in Enum.GetValues(typeof(EvaluationMetricEnum)).Cast<EvaluationMetricEnum>())
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
                   +" - "+ Model.MetricsToString(TrainingMetrics, "Training")
                   +" - "+ Model.MetricsToString(ValidationMetrics, "Validation");
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
        public double GetTrainingLoss(EvaluationMetricEnum evaluationMetric) => GetWithDefaultValue(TrainingMetrics, evaluationMetric, double.NaN);
        public double GetValidationLoss(EvaluationMetricEnum evaluationMetric) => GetWithDefaultValue(ValidationMetrics, evaluationMetric, double.NaN);
        // ReSharper disable once UnusedMember.Global
        public double TrainingAccuracy => GetWithDefaultValue(TrainingMetrics, EvaluationMetricEnum.Accuracy, double.NaN);
        public double ValidationAccuracy => GetWithDefaultValue(ValidationMetrics, EvaluationMetricEnum.Accuracy, double.NaN);
        // ReSharper disable once UnusedMember.Global
        public double TrainingMae => GetWithDefaultValue(TrainingMetrics, EvaluationMetricEnum.Mae, double.NaN);
        // ReSharper disable once UnusedMember.Global
        public double ValidationMae => GetWithDefaultValue(ValidationMetrics, EvaluationMetricEnum.Mae, double.NaN);

        private static double GetWithDefaultValue(IDictionary<EvaluationMetricEnum, double> allAvailableMetrics, EvaluationMetricEnum metricEnum, double defaultValue)
        {
            return allAvailableMetrics.ContainsKey(metricEnum)
                ? allAvailableMetrics[metricEnum]
                : defaultValue;
        }

        /// <summary>
        /// used only for tests
        /// </summary>
        private static IDictionary<EvaluationMetricEnum, double> ToDictionary(double loss, double accuracy, EvaluationMetricEnum evaluationMetric)
        {
            return new Dictionary<EvaluationMetricEnum, double>
            {
                [evaluationMetric] = loss,
                [EvaluationMetricEnum.Accuracy] = accuracy
            };
        }
    }
}
