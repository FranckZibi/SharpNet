//#define PROFILE_MEMORY_POOL
using System;
using System.Collections.Generic;
using System.Diagnostics;
#if PROFILE_MEMORY_POOL
using System.Diagnostics;
#endif
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
#if PROFILE_MEMORY_POOL
        private readonly Stopwatch _sw = new Stopwatch();
#endif
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
            FreeMemory(t);
            t = null;
        }

        public void FreeMemory(Tensor t)
        {
            if (t == null)
            {
                return;
            }
#if PROFILE_MEMORY_POOL
            _sw.Start();
#endif
            _availableTensorOrderedByCount.Add(t);
            SortAvailableTensorsByCapacity();
#if PROFILE_MEMORY_POOL
            _sw.Stop();
#endif
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
            var result = (IsMock ? "Mock" : "")+"Used MemoryPool: " + Utils.MemoryBytesToString(CapacityInBytes);
#if PROFILE_MEMORY_POOL
            result += " " + _sw.ElapsedMilliseconds + "ms";
#endif
            return result;
        }
        /// <summary>
        /// return a buffer with a minimal capacity of 'minimalSizeInBytes' bytes
        /// </summary>
        /// <param name="minimalSizeInBytes">the minimal size in bytes of the buffer</param>
        /// <param name="description"></param>
        /// <returns></returns>
        public Tensor GetBuffer(size_t minimalSizeInBytes, string description)
        {
            Tensor buffer = null;
            GetBuffer(ref buffer, minimalSizeInBytes, description);
            return buffer;
        }
        public void GetBuffer(ref Tensor buffer, size_t minimalSizeInBytes, string description)
        {
            var count = (minimalSizeInBytes + sizeof(float) - 1) / sizeof(float);
            count = Math.Max(count, 1);
            GetNotInitializedFloatTensor(ref buffer, new[] { (int)count }, description);
        }
        /// <summary>
        /// return a new float tensor .
        /// the data in the tensor is not initialized
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="description"></param>
        public Tensor GetNotInitializedFloatTensor(int[] shape, string description)
        {
#if PROFILE_MEMORY_POOL
            _sw.Start();
#endif
            var neededMemoryInBytes = NeededMemoryInBytes(shape);
            for (var i = 0; i < _availableTensorOrderedByCount.Count; i++)
            {
                var availableTensor = _availableTensorOrderedByCount[i];
                if (availableTensor.CapacityInBytes >= neededMemoryInBytes)
                {
                    //the smallest tensor with enough capacity to store our tensor is 10x too big
                    //we prefer to allocate a new smaller tensor then to waste a big tensor
                    if (availableTensor.CapacityInBytes > Math.Max(10 * neededMemoryInBytes, 10*1000))
                    {
                        break;
                    }

                    _availableTensorOrderedByCount.RemoveAt(i);
                    availableTensor.Reshape(shape);
                    availableTensor.Description = description;
#if PROFILE_MEMORY_POOL
                    _sw.Stop();
#endif
                    return availableTensor;
                }
            }
            //no available tensor found, we need to allocate a new tensor
            var newlyAllocatedTensor = AllocateNewTensor(shape, description);
#if PROFILE_MEMORY_POOL
            _sw.Stop();
#endif
            return newlyAllocatedTensor;
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
            Debug.Assert(shape != null);
            if (bufferIfAny != null && string.IsNullOrEmpty(description))
            {
                description = bufferIfAny.Description;
            }
            if (bufferIfAny == null)
            {
                bufferIfAny = GetNotInitializedFloatTensor(shape, description);
                return;
            }
            if (bufferIfAny.HasEnoughCapacityForTensor(shape))
            {
                bufferIfAny.Reshape(shape);
                return;
            }
            FreeMemory(ref bufferIfAny);
            bufferIfAny = GetNotInitializedFloatTensor(shape, description);
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

        private Tensor AllocateNewTensor(int[] shape, string description)
        {
            Debug.Assert(shape != null);
            Tensor result;
            if (IsMock)
            {
                result = new MockTensor<float>(shape, description);
            }
            else if (_gpuWrapper != null)
            {
                result = new GPUTensor<float>(shape, null, _gpuWrapper, description);
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
        private void SortAvailableTensorsByCapacity()
        {
            _availableTensorOrderedByCount.Sort((x, y) => (int)(x.CapacityInBytes - y.CapacityInBytes));
        }
    }
}
