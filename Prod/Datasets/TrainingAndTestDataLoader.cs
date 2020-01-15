namespace SharpNet.Datasets
{
    public class TrainingAndTestDataLoader : AbstractTrainingAndTestDataSet
    {
        public TrainingAndTestDataLoader(IDataSet training, IDataSet test, AbstractDataSet parent)
            : base(parent.Name, parent.Channels, parent.Height, parent.Width, parent.Categories)
        {
            Training = training;
            Test = test;
        }

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }
    }
}