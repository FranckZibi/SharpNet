using System;
using System.Collections.Generic;

namespace SharpNet.Data
{
    public class EpochData
    {
        #region properties
        public int Index { get; }
        public double LearningRateFromScheduler { get; }
        public double LearningRateMultiplicativeFactorFromReduceLrOnPlateau { get; }
        public double TrainingLoss { get; set; }
        public double TrainingAccuracy { get; set; }
        public double ValidationLoss { get; set; }
        public double ValidationAccuracy { get; set; }
        public double SecondsForEpoch { get; set; }
        #endregion

        public EpochData(int index, double learningRateFromScheduler, double learningRateMultiplicativeFactorFromReduceLROnPlateau, double trainingLoss, double trainingAccuracy, double validationLoss, double validationAccuracy)
        {
            Index = index;
            LearningRateFromScheduler = learningRateFromScheduler;
            LearningRateMultiplicativeFactorFromReduceLrOnPlateau = learningRateMultiplicativeFactorFromReduceLROnPlateau;
            TrainingLoss = trainingLoss;
            TrainingAccuracy = trainingAccuracy;
            ValidationLoss = validationLoss;
            ValidationAccuracy = validationAccuracy;
        }
        public double LearningRate => LearningRateFromScheduler * LearningRateMultiplicativeFactorFromReduceLrOnPlateau;

        #region serialization
        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(Index), Index)
                .Add(nameof(LearningRateFromScheduler), LearningRateFromScheduler)
                .Add(nameof(LearningRateMultiplicativeFactorFromReduceLrOnPlateau), LearningRateMultiplicativeFactorFromReduceLrOnPlateau)
                .Add(nameof(TrainingLoss), TrainingLoss).Add(nameof(TrainingAccuracy), TrainingAccuracy)
                .Add(nameof(ValidationLoss), ValidationLoss).Add(nameof(ValidationAccuracy), ValidationAccuracy)
                .Add(nameof(SecondsForEpoch), SecondsForEpoch)
                .ToString();
        }
        public EpochData(IDictionary<string, object> serialized)
        {
            Index = (int)serialized[nameof(Index)];
            LearningRateFromScheduler = (double)serialized[nameof(LearningRateFromScheduler)];
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
            return "Epoch "+Index+" : learningRate:"+LearningRate+ "; TrainingLoss:"+TrainingLoss+ "; TrainingAccuracy:" + TrainingAccuracy + "; ValidationLoss:" + ValidationLoss + "; ValidationAccuracy:" + ValidationAccuracy;
        }
        public bool Equals(EpochData other, double epsilon)
        {
            return Index == other.Index
                   && Math.Abs(LearningRateFromScheduler - other.LearningRateFromScheduler) <= epsilon
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
            return Index == ((EpochData)obj).Index;
        }
        public override int GetHashCode()
        {
            return Index;
        }
    }
}
