using System;

namespace SharpNet.Datasets
{
    public interface ITrainingAndTestDataSet : IDisposable
    {
        IDataSet Training { get; }
        IDataSet Test { get; }

        string Name { get; }

        int CategoryByteToCategoryIndex(byte categoryByte);
        byte CategoryIndexToCategoryByte(int categoryIndex);
    }
}