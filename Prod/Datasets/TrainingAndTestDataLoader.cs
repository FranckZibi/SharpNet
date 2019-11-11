namespace SharpNet.Datasets
{
    public class TrainingAndTestDataLoader : ITrainingAndTestDataSet
    {
        public TrainingAndTestDataLoader(IDataSet training, IDataSet test, string name)
        {
            Training = training;
            Test = test;
            Name = name;
        }

        public void Dispose()
        {
            Training.Dispose();
            Test.Dispose();
        }

        public IDataSet Training { get; }
        public IDataSet Test { get; }
        public string Name { get; }
    }
}