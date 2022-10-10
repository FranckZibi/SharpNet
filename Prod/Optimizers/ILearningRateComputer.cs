using System.Collections.Generic;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public interface ILearningRateComputer
    {
        double LearningRate(int epoch, double percentagePerformedInEpoch, double learningRateMultiplicativeFactorFromReduceLrOnPlateau);
        bool ShouldReduceLrOnPlateau(List<EpochData> previousEpochsData, EvaluationMetricEnum loss);
        double MultiplicativeFactorFromReduceLrOnPlateau(List<EpochData> previousEpochsData, EvaluationMetricEnum loss);
        bool ShouldCreateSnapshotForEpoch(int epoch);
        double MaxLearningRate { get;  }

    }
}
