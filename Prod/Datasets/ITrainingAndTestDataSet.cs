using System;

namespace SharpNet.Datasets
{
    public interface ITrainingAndTestDataSet : IDisposable
    {
        IDataSet Training { get; }
        IDataSet Test { get; }
    }
}
