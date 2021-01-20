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
#if PROFILE_MEMORY_POOL
        private readonly Stopwatch _sw = new Stopwatch();
#endif
        private bool _disposed = false;
        private readonly GPUWrapper _gpuWrapper;
        #endregion

        #region public properties
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
        #endregion

        #region constructor
        public TensorMemoryPool(GPUWrapper gpuWrapper, bool isMock)
        {
            _gpuWrapper = gpuWrapper;
            IsMock = isMock;
        }
        #endregion

        public override string ToString()
        {
            var result = (IsMock ? "Mock" : "")+"Used MemoryPool: " + Utils.MemoryBytesToString(CapacityInBytes);
#if PROFILE_MEMORY_POOL
            result += " " + _sw.ElapsedMilliseconds + "ms";
#endif
            return result;
        }

        /// <summary>
        /// return a new float tensor .
        /// the data in the tensor is not initialized
        /// </summary>
        /// <param name="shape"></param>
        public Tensor GetFloatTensor(int[] shape)
        {
#if PROFILE_MEMORY_POOL
            _sw.Start();
#endif
            var neededMemoryInBytes = (ulong)(Utils.Product(shape) * sizeof(float));
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
#if PROFILE_MEMORY_POOL
                    _sw.Stop();
#endif
                    return availableTensor;
                }
            }
            //no available tensor found, we need to allocate a new tensor
            var newlyAllocatedTensor = AllocateNewTensor(shape);
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
        public void GetFloatTensor(ref Tensor bufferIfAny, int[] shape)
        {
            Debug.Assert(shape != null);
            if (bufferIfAny == null)
            {
                bufferIfAny = GetFloatTensor(shape);
                return;
            }
            if (bufferIfAny.HasEnoughCapacityForTensor(shape))
            {
                bufferIfAny.Reshape(shape);
                return;
            }
            FreeFloatTensor(ref bufferIfAny);
            bufferIfAny = GetFloatTensor(shape);
        }

        /// <summary>
        /// return a buffer with a minimal capacity of 'minimalSizeInBytes' bytes
        /// </summary>
        /// <param name="minimalSizeInBytes">the minimal size in bytes of the buffer</param>
        /// <returns></returns>
        public Tensor GetBuffer(size_t minimalSizeInBytes)
        {
            Tensor buffer = null;
            GetBuffer(ref buffer, minimalSizeInBytes);
            return buffer;
        }
        public void GetBuffer(ref Tensor buffer, size_t minimalSizeInBytes)
        {
            var count = (minimalSizeInBytes + sizeof(float) - 1) / sizeof(float);
            count = Math.Max(count, 1);
            GetFloatTensor(ref buffer, new[] { (int)count });
        }

        public void FreeFloatTensor(ref Tensor t)
        {
            FreeFloatTensor(t);
            t = null;
        }
        public void FreeFloatTensor(Tensor t)
        {
            if (t == null || !t.IsOwnerOfMemory)
            {
                return;
            }
#if PROFILE_MEMORY_POOL
            _sw.Start();
#endif

#if DEBUG
            for (int i = 0; i < _availableTensorOrderedByCount.Count; ++i)
            {
                if ( !(t is MockTensor<float>) && t.Pointer == _availableTensorOrderedByCount[i].Pointer)
                {
                    throw new Exception("object already available with pointer "+ t.Pointer+" and can't be freed twice ("+t+")");
                }
            }
#endif
            _availableTensorOrderedByCount.Add(t);
            _availableTensorOrderedByCount.Sort((x, y) => (int)(x.CapacityInBytes - y.CapacityInBytes));
#if PROFILE_MEMORY_POOL
            _sw.Stop();
#endif
        }
        public void FreeFloatTensor(IList<Tensor> list, int index)
        {
            var t = list[index];
            FreeFloatTensor(ref t);
            list[index] = null;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _allAllocatedTensors.ForEach(t => t.Dispose());
            _allAllocatedTensors.Clear();
            _availableTensorOrderedByCount.Clear();
        }
        
        private Tensor AllocateNewTensor(int[] shape)
        {
            Debug.Assert(shape != null);
            Tensor result;
            if (IsMock)
            {
                result = new MockTensor<float>(shape);
            }
            else if (_gpuWrapper != null)
            {
                result = new GPUTensor<float>(shape, null, _gpuWrapper);
            }
            else
            {
                result = new CpuTensor<float>(shape, null);
            }
            _allAllocatedTensors.Add(result);
            return result;
        }
    }
}
