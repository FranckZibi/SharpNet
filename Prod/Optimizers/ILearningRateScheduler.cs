namespace SharpNet.Optimizers
{
    public interface ILearningRateScheduler
    {
        double LearningRate(int epoch, int miniBatchBlockIdInEpoch, int nbBlocksInEpoch);

        bool ShouldCreateSnapshotForEpoch(int epoch);
    }
}