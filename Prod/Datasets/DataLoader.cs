namespace SharpNet.Datasets
{
    public class DataLoader : IDataSet
    {
        public DataLoader(IDataSetLoader training, IDataSetLoader test)
        {
            Training = training;
            Test = test;
        }

        public void Dispose()
        {
            Training.Dispose();
            Test.Dispose();
        }

        public IDataSetLoader Training { get; }
        public IDataSetLoader Test { get; }
    }
}