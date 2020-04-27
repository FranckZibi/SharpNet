namespace SharpNet.Optimizers
{
    public interface ILearningRateScheduler
    {
        double LearningRate(int epoch, double percentagePerformedInEpoch);

        bool ShouldCreateSnapshotForEpoch(int epoch);
    }
}