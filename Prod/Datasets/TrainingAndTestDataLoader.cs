namespace SharpNet.Datasets
{
    public class TrainingAndTestDataLoader : AbstractTrainingAndTestDataSet
    {
        public TrainingAndTestDataLoader(IDataSet training, IDataSet test, AbstractDataSet parent)
            : base(parent.Name, parent.CategoryCount)
        {
            Training = training;
            Test = test;
        }

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }
    }
}