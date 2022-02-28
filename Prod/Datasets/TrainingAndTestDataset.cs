namespace SharpNet.Datasets
{
    public class TrainingAndTestDataset : AbstractTrainingAndTestDataset
    {
        public TrainingAndTestDataset(IDataSet training, IDataSet test, string name)
            : base(name)
        {
            Training = training;
            Test = test;
        }

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }
    }
}