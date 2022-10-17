namespace SharpNet.Datasets
{
    public abstract class AbstractTrainingAndTestDataset : ITrainingAndTestDataSet
    {
        #region public properties
        public abstract DataSet Training { get; }
        public abstract DataSet Test { get; }
        public string Name { get; }
        #endregion

        #region constructor
        protected AbstractTrainingAndTestDataset(string name)
        {
            Name = name;
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
