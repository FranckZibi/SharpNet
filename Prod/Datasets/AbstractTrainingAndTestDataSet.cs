namespace SharpNet.Datasets
{
    public abstract class AbstractTrainingAndTestDataSet : ITrainingAndTestDataSet
    {
        public abstract IDataSet Training { get; }
        public abstract IDataSet Test { get; }

        public string Name { get; }
        public int Channels { get; }
        public int Height { get; }
        public int Width { get; }
        public int CategoryCount { get; }
        public int[] InputShape_CHW => new[]{ Channels, Height, Width };


        protected AbstractTrainingAndTestDataSet(string name, int channels, int height, int width, int categoryCount)
        {
            Name = name;
            Channels = channels;
            Height = height;
            Width = width;
            CategoryCount = categoryCount;
        }

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