using System;
using SharpNet.Data;

namespace SharpNet.GPU
{
    /// <summary>
    /// wrapper to memory located in the device (= GPU)
    /// </summary>
    public sealed class DeviceMemory : IDisposable
    {
        #region private fields
        private IntPtr _pointer;
        private bool _disposed;
        #endregion
        #region readonly properties
        private readonly size_t _capacityInBytes;
        #endregion

        /// <summary>
        /// Allocate memory on the device.
        /// This memory will need to be free when disposing the current object  
        /// </summary>
        /// <param name="capacityInBytes">size in byte to allocate on device</param>
        public DeviceMemory(size_t capacityInBytes)
        {
            _capacityInBytes = capacityInBytes;
            var res = NVCudaWrapper.cuMemAlloc_v2(out _pointer, _capacityInBytes);
            GPUWrapper.CheckStatus(res);
            IsOwnerOfDeviceMemory = true;
        }
        /// <summary>
        /// Pointer to an already allocated memory on device.
        /// This memory area should not be freed when disposing the current object
        /// </summary>
        /// <param name="pointer">address of memory on device</param>
        /// <param name="capacityInBytes">size in bytes already allocated on device</param>
        public DeviceMemory(IntPtr pointer, size_t capacityInBytes)
        {
            _capacityInBytes = capacityInBytes;
            _pointer = pointer;
            IsOwnerOfDeviceMemory = false;
        }

        public IntPtr Pointer
        {
            get
            {
                AssertIsNotDisposed();
                return _pointer;
            }
        }

        public bool IsOwnerOfDeviceMemory { get; }

        public void AssertIsNotDisposed()
        {
            if (_disposed)
            {
                throw new Exception("disposed object of size " + _capacityInBytes + " / IsOwnerOfDeviceMemory=" + IsOwnerOfDeviceMemory);
            }
        }

        public void ZeroMemory()
        {
            AssertIsNotDisposed();
            CUresult res;
            if (_capacityInBytes % 4 == 0)
            {
                res = NVCudaWrapper.cuMemsetD32_v2(_pointer, 0, _capacityInBytes / 4);
            }
            else
            {
                res = NVCudaWrapper.cuMemsetD8_v2(_pointer, (char)0, _capacityInBytes);
            }
            GPUWrapper.CheckStatus(res);
        }

        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        ~DeviceMemory()
        {
            if (!IsOwnerOfDeviceMemory)
            {
                return;
            }
            Dispose(false);
        }
        // ReSharper disable once UnusedParameter.Local
        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            //unmanaged memory
            if (IsOwnerOfDeviceMemory)
            {
                /*var res =*/ NVCudaWrapper.cuMemFree_v2(_pointer);
                //GPUWrapper.CheckStatus(res);
                _pointer = IntPtr.Zero;
            }
        }
        #endregion

    }
}
