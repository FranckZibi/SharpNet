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
        private readonly IntPtr _pointer;
        private readonly size_t _capacityInBytes;
        private bool _disposed;
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
        }
        public IntPtr Pointer
        {
            get
            {
                AssertIsNotDisposed();
                return _pointer;
            }
        }

        public void AssertIsNotDisposed()
        {
            if (_disposed)
            {
                throw new Exception("disposed object of size " + _capacityInBytes);
            }
        }

        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        ~DeviceMemory()
        {
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
            if (disposing)
            {
                //managed memory
                //nothing to do
            }
            //unmanaged memory
            /*var res =*/ NVCudaWrapper.cuMemFree_v2(_pointer);
            //GPUWrapper.CheckStatus(res);
        }
        #endregion
    }
}
