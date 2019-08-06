using System;
using SharpNet.CPU;
using SharpNet.Data;

namespace SharpNet.Datasets
{
    public interface IDataSetLoader<T> : IDisposable where T: struct
    {
        /// <summary>
        /// Load 'elementCount' elements from the one starting at index 'indexFirstElement' and for epoch 'numEpoch'
        /// </summary>
        /// <param name="epoch">index of epoch. The first epoch is 1</param>
        /// <param name="isTraining"></param>
        /// <param name="indexFirstElement">The index of the first element to load (the very first element fo the data set is at index 0</param>
        /// <param name="miniBatchSize">number of elements to load (from the one art index 'indexFirstElement')</param>
        /// <param name="randomizeOrder"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        void Load(int epoch, bool isTraining, int indexFirstElement, int miniBatchSize, bool randomizeOrder, ref Tensor x, ref Tensor y);
        /// <summary>
        /// number of elements in DataSet
        /// </summary>
        int Count { get; }

        int TypeSize { get; }

        /// <summary>
        ///  number of distinct categories in the DataSet
        /// </summary>
        int Categories { get; }
        /// <summary>
        /// number of channels of each elements
        /// </summary>
        int Channels { get; }
        /// <summary>
        /// Current height of elements to load
        /// </summary>
        int CurrentHeight { get; }
        /// <summary>
        /// Current width of elements to load
        /// </summary>
        int CurrentWidth { get; }

        IDataSetLoader<float> ToSinglePrecision();
        IDataSetLoader<double> ToDoublePrecision();

        int[] Y_Shape { get; }
        int[] XChunk_Shape(int miniBatchSize);
        int[] YChunk_Shape(int miniBatchSize);

        CpuTensor<T> Y { get; }

    }
}