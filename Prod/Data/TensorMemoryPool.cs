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

        public void FreeMemory(Tensor t)
        {
            _availableTensorOrderedByCount.Add(t);
            _availableTensorOrderedByCount.Sort((x, y) => (int)(x.CapacityInBytes - y.CapacityInBytes));
        }
        public void FreeMemory(IList<Tensor> list, int index)
        {
            var t = list[index];
            if (t != null)
            {
                FreeMemory(t);
                list[index] = null;
            }
        }
        public void FreeMemory(IList<Tensor> list)
        {
            foreach (var t in list)
            {
                if (t != null)
                {
                    FreeMemory(t);
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
            string result = IsMock?"Mock":"";
            result += "MemoryPool Used: " + Utils.MemoryBytesToString(ReallyNeededMemoryInBytes) +"/"+ Utils.MemoryBytesToString(CapacityInBytes);
            return result;
        }
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
            var newTensor = AllocateNewTensor(shape, description);
            return newTensor;
        }
        public Tensor GetNotInitializedFloatTensor(int[] shape, Tensor bufferIfAny, string description)
        {
            if (bufferIfAny == null)
            {
                return GetNotInitializedFloatTensor(shape, description);
            }
            if (bufferIfAny.HasEnoughCapacityForTensor(shape))
            {
                bufferIfAny.Reshape(shape);
                return bufferIfAny;
            }
            FreeMemory(bufferIfAny);
            return GetNotInitializedFloatTensor(shape, description);
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
                result = new GPUTensor<float>(shape, description, _gpuWrapper);
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
        private ulong ReallyNeededMemoryInBytes
        {
            get
            {
                var allReallyNeededMemoryInBytes = (_allAllocatedTensors.Count == 0) ? 0 : _allAllocatedTensors.Select(t => t.ReallyNeededMemoryInBytes).Sum();
                var availableReallyNeededMemoryInBytes = (_availableTensorOrderedByCount.Count == 0) ? 0 : _availableTensorOrderedByCount.Select(t => t.ReallyNeededMemoryInBytes).Sum();
                return allReallyNeededMemoryInBytes - availableReallyNeededMemoryInBytes;
            }
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