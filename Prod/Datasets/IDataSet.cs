using System;

namespace SharpNet.Datasets
{
    public interface IDataSet : IDisposable
    {
        IDataSetLoader Training { get; }
        IDataSetLoader Test { get; }

        string Name { get; }
    }
}