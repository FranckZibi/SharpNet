namespace SharpNet.Datasets
{
    public class TrainingAndTestDataLoader : AbstractTrainingAndTestDataSet
    {
        public TrainingAndTestDataLoader(IDataSet training, IDataSet test, string name)
            : base(name)
        {
            Training = training;
            Test = test;
        }

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }
    }
}