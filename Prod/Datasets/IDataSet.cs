using System;

namespace SharpNet.Datasets
{
    public interface IDataSet<T> : IDisposable where T : struct
    {
        IDataSetLoader<T> Training { get; }
        IDataSetLoader<T> Test { get; }
    }
}