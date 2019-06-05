namespace SharpNet.Optimizers
{
    public interface ILearningRateScheduler
    {
        double LearningRate(int epoch, int blockIdInEpoch, int nbBlocksInEpoch);

        bool ShouldCreateSnapshotForEpoch(int epoch);
    }
}