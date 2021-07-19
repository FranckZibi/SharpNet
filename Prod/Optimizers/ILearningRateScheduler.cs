namespace SharpNet.Optimizers
{
    public interface ILearningRateScheduler
    {
        double LearningRate(int epoch, double percentagePerformedInEpoch);

        /// <summary>
        /// the maximum possible value of the learning rate
        /// </summary>
        double MaxLearningRate { get; }

        bool ShouldCreateSnapshotForEpoch(int epoch);
    }
}