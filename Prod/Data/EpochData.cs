using System;
using System.Collections.Generic;

namespace SharpNet.Data
{
    public class EpochData
    {
        #region properties
        private readonly int _index;
        public double LearningRateMultiplicativeFactorFromReduceLrOnPlateau { get; }
        private double LearningRateAtEpochStart { get; }
        public double TrainingLoss { get; }
        private double TrainingAccuracy { get; }
        private double ValidationLoss { get; }
        private double ValidationAccuracy { get; }
        private double SecondsForEpoch { get; }
        #endregion

        public EpochData(int index, double learningRateAtEpochStart, double learningRateMultiplicativeFactorFromReduceLROnPlateau, double trainingLoss, double trainingAccuracy, double validationLoss, double validationAccuracy, double secondsForEpoch)
        {
            _index = index;
            LearningRateAtEpochStart = learningRateAtEpochStart;
            LearningRateMultiplicativeFactorFromReduceLrOnPlateau = learningRateMultiplicativeFactorFromReduceLROnPlateau;
            TrainingLoss = trainingLoss;
            TrainingAccuracy = trainingAccuracy;
            ValidationLoss = validationLoss;
            ValidationAccuracy = validationAccuracy;
            SecondsForEpoch = secondsForEpoch;
        }
        #region serialization
        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(_index), _index)
                .Add(nameof(LearningRateAtEpochStart), LearningRateAtEpochStart)
                .Add(nameof(LearningRateMultiplicativeFactorFromReduceLrOnPlateau), LearningRateMultiplicativeFactorFromReduceLrOnPlateau)
                .Add(nameof(TrainingLoss), TrainingLoss).Add(nameof(TrainingAccuracy), TrainingAccuracy)
                .Add(nameof(ValidationLoss), ValidationLoss).Add(nameof(ValidationAccuracy), ValidationAccuracy)
                .Add(nameof(SecondsForEpoch), SecondsForEpoch)
                .ToString();
        }
        public EpochData(IDictionary<string, object> serialized)
        {
            _index = (int)serialized[nameof(_index)];
            LearningRateAtEpochStart = (double)serialized[nameof(LearningRateAtEpochStart)];
            LearningRateMultiplicativeFactorFromReduceLrOnPlateau = (double)serialized[nameof(LearningRateMultiplicativeFactorFromReduceLrOnPlateau)];
            TrainingLoss = (double)serialized[nameof(TrainingLoss)];
            TrainingAccuracy = (double)serialized[nameof(TrainingAccuracy)];
            ValidationLoss = (double)serialized[nameof(ValidationLoss)];
            ValidationAccuracy = (double)serialized[nameof(ValidationAccuracy)];
            SecondsForEpoch = (double)serialized[nameof(SecondsForEpoch)];
        }
        #endregion

        public override string ToString()
        {
            return "Epoch "+_index+" : learningRate:"+LearningRateAtEpochStart+ "; TrainingLoss:"+TrainingLoss+ "; TrainingAccuracy:" + TrainingAccuracy + "; ValidationLoss:" + ValidationLoss + "; ValidationAccuracy:" + ValidationAccuracy;
        }
        public bool Equals(EpochData other, double epsilon)
        {
            return _index == other._index
                   && Math.Abs(LearningRateAtEpochStart - other.LearningRateAtEpochStart) <= epsilon
                   && Math.Abs(LearningRateMultiplicativeFactorFromReduceLrOnPlateau - other.LearningRateMultiplicativeFactorFromReduceLrOnPlateau) <= epsilon
                   && Math.Abs(TrainingLoss - other.TrainingLoss) <= epsilon
                   && Math.Abs(TrainingAccuracy - other.TrainingAccuracy) <= epsilon
                   && Math.Abs(ValidationLoss - other.ValidationLoss) <= epsilon
                   && Math.Abs(ValidationAccuracy - other.ValidationAccuracy) <= epsilon
                   && Math.Abs(SecondsForEpoch - other.SecondsForEpoch) <= epsilon
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
    }
}
