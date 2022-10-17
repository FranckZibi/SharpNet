using System;

namespace SharpNet.Datasets
{
    public interface ITrainingAndTestDataSet : IDisposable
    {
        DataSet Training { get; }
        DataSet Test { get; }
    }
}
