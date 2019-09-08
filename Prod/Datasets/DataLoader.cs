namespace SharpNet.Datasets
{
    public class DataLoader : IDataSet
    {
        public DataLoader(IDataSetLoader training, IDataSetLoader test, string name)
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

        public IDataSetLoader Training { get; }
        public IDataSetLoader Test { get; }
        public string Name { get; }
    }
}