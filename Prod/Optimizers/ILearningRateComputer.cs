using System.Collections.Generic;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public interface ILearningRateComputer
    {
        double LearningRate(int epoch, int blockIdInEpoch, int nbBlocksInEpoch, double learningRateMultiplicativeFactorFromReduceLrOnPlateau);
        bool ShouldReduceLrOnPlateau(List<EpochData> previousEpochsData);
        double MultiplicativeFactorFromReduceLrOnPlateau(List<EpochData> previousEpochsData);
        bool ShouldCreateSnapshotForEpoch(int epoch);
    }
}
