using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.CPU;
using SharpNet.GPU;

namespace SharpNet.Data
{
    public class TensorMemoryPool : IDisposable
    {
        #region private fields
        private readonly List<Tensor> _availableTensorOrderedByCount = new List<Tensor>();
        private readonly List<Tensor> _allAllocatedTensors = new List<Tensor>();
        //null if we should allocate CpuTensor
        //not  null for GPUTensor
        private readonly GPUWrapper _gpuWrapper;
        private bool _disposed = false;
        #endregion

        public ulong CapacityInBytes
        {
            get
            {
                if (_allAllocatedTensors.Count == 0)
                {
                    return 0;
                }
                return _allAllocatedTensors.Select(t => t.CapacityInBytes).Sum();
            }
        }
        public bool IsMock { get; }

        public TensorMemoryPool(GPUWrapper gpuWrapper, bool isMock)
        {
            _gpuWrapper = gpuWrapper;
            IsMock = isMock;
        }

        public void FreeMemory(ref Tensor t)
        {
            if (t == null)
            {
                return;
            }
            _availableTensorOrderedByCount.Add(t);
            SortAvailableTensorsByCapacity();
            t = null;
        }

        private void SortAvailableTensorsByCapacity()
        {
            _availableTensorOrderedByCount.Sort((x, y) => (int)(x.CapacityInBytes - y.CapacityInBytes));
        }

        public void FreeMemory(IList<Tensor> list, int index)
        {
            var t = list[index];
            FreeMemory(ref t);
            list[index] = null;
        }
        public void FreeMemory(IList<Tensor> list)
        {
            for (var index = 0; index < list.Count; index++)
            {
                var t = list[index];
                if (t != null)
                {
                    FreeMemory(ref t);
                }
            }
            list.Clear();
        }
        public override string ToString()
        {
            return MemoryInfo();
        }
        public string MemoryInfo()
        {
            return (IsMock ? "Mock" : "")+"Used MemoryPool: " + Utils.MemoryBytesToString(CapacityInBytes);
        }

        /// <summary>
        /// return a buffer with a minimal capacity of 'minimalSizeInBytes' bytes
        /// </summary>
        /// <param name="minimalSizeInBytes">the minimal size in bytes of the buffer</param>
        /// <returns></returns>
        public Tensor GetBuffer(size_t minimalSizeInBytes)
        {
            var count = (minimalSizeInBytes + sizeof(float) - 1) / sizeof(float);
            count = Math.Max(count, 1);
            return GetNotInitializedFloatTensor(new[] {(int)count}, "buffer");
        }

        /// <summary>
        /// return a new float tensor .
        /// the data in the tensor is not initialized
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="description"></param>
        public Tensor GetNotInitializedFloatTensor(int[] shape, string description)
        {
            var neededMemoryInBytes = NeededMemoryInBytes(shape);
            for (var i = 0; i < _availableTensorOrderedByCount.Count; i++)
            {
                var availableTensor = _availableTensorOrderedByCount[i];
                if (availableTensor.CapacityInBytes >= neededMemoryInBytes)
                {
                    _availableTensorOrderedByCount.RemoveAt(i);
                    availableTensor.Reshape(shape);
                    availableTensor.Description = description;
                    return availableTensor;
                }
            }
            //no available tensor found, we need to allocate a new tensor
            return AllocateNewTensor(shape, description);
        }
        /// <summary>
        /// return a new float tensor .
        /// the data in the tensor is not initialized
        /// </summary>
        /// <param name="bufferIfAny"></param>
        /// <param name="shape"></param>
        /// <param name="description"></param>
        public void GetNotInitializedFloatTensor(ref Tensor bufferIfAny, int[] shape, string description = "")
        {
            if (bufferIfAny != null && string.IsNullOrEmpty(description))
            {
                description = bufferIfAny.Description;
            }
            if (bufferIfAny == null)
            {
                bufferIfAny = GetNotInitializedFloatTensor(shape, description);
            }
            else if (bufferIfAny.HasEnoughCapacityForTensor(shape))
            {
                bufferIfAny.Reshape(shape);
            }
            else
            {
                FreeMemory(ref bufferIfAny);
                bufferIfAny = GetNotInitializedFloatTensor(shape, description);
            }
        }

       


        private Tensor AllocateNewTensor(int[] shape, string description)
        {
            Tensor result;
            if (IsMock)
            {
                result = new MockTensor<float>(shape, description);
            }
            else if (_gpuWrapper != null)
            {
                result = new GPUTensor<float>(shape, null, description, _gpuWrapper);
            }
            else
            {
                result = new CpuTensor<float>(shape, null, description);
            }
            _allAllocatedTensors.Add(result);
            return result;
        }

        private static ulong NeededMemoryInBytes(int[] shape)
        {
            return (ulong)(Utils.Product(shape) * sizeof(float));
        }

        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            if (disposing)
            {
                //managed memory
                _allAllocatedTensors.ForEach(t => t.Dispose());
                _allAllocatedTensors.Clear();
                _availableTensorOrderedByCount.Clear();
            }
            //unmanaged memory
        }
        ~TensorMemoryPool()
        {
            Dispose(false);
        }
        #endregion
    }
}