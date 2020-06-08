namespace SharpNet.Datasets
{
    public abstract class AbstractTrainingAndTestDataSet : ITrainingAndTestDataSet
    {
        #region public properties
        public abstract IDataSet Training { get; }
        public abstract IDataSet Test { get; }
        public string Name { get; }
        public int CategoryCount { get; }
        #endregion

        #region constructor
        protected AbstractTrainingAndTestDataSet(string name, int categoryCount)
        {
            Name = name;
            CategoryCount = categoryCount;
        }
        #endregion

        public virtual void Dispose()
        {
            Training?.Dispose();
            Test?.Dispose();
        }
        public virtual int CategoryByteToCategoryIndex(byte categoryByte)
        {
            return categoryByte;
        }
        public virtual byte CategoryIndexToCategoryByte(int categoryIndex)
        {
            return (byte)categoryIndex;
        }
    }
}
