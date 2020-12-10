namespace SharpNet.Datasets
{
    public class TrainingAndTestDataLoader : AbstractTrainingAndTestDataSet
    {
        public TrainingAndTestDataLoader(IDataSet training, IDataSet test, IDataSet parent)
            : base(parent.Name)
        {
            Training = training;
            Test = test;
        }

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }
    }
}