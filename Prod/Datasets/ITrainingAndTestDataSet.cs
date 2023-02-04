using System;
namespace SharpNet.Datasets
{
    public interface ITrainingAndTestDataset : IDisposable
    {
        DataSet Training { get; }
        DataSet Test { get; }
    }
}
