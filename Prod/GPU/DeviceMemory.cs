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
        private readonly bool _isOwnerOfDeviceMemory;
        private bool _disposed;
        #endregion
        #region readonly properties
        public size_t SizeInBytes { get; }
        #endregion

        /// <summary>
        /// Allocate memory on the device.
        /// This memory will need to be free when disposing the current object  
        /// </summary>
        /// <param name="sizeInBytes">size in byte to allocate on device</param>
        public DeviceMemory(size_t sizeInBytes)
        {
            SizeInBytes = sizeInBytes;
            var res = NVCudaWrapper.cuMemAlloc_v2(out _pointer, SizeInBytes);
            GPUWrapper.CheckStatus(res);
            _isOwnerOfDeviceMemory = true;
        }
        /// <summary>
        /// Pointer to an already allocated memory on device.
        /// This memory area should not be freed when disposing the current object
        /// </summary>
        /// <param name="pointer">adress of memory on device</param>
        /// <param name="sizeInBytes">size in bytes already allocated on device</param>
        public DeviceMemory(IntPtr pointer, size_t sizeInBytes)
        {
            SizeInBytes = sizeInBytes;
            _pointer = pointer;
            _isOwnerOfDeviceMemory = false;
        }
        
        public IntPtr Pointer => _pointer;
        public void ZeroMemory()
        {
            CUresult res;
            if (SizeInBytes % 4 == 0)
            {
                res = NVCudaWrapper.cuMemsetD32_v2(_pointer, 0, SizeInBytes / 4);
            }
            else
            {
                res = NVCudaWrapper.cuMemsetD8_v2(_pointer, (char)0, SizeInBytes);
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
            if (_isOwnerOfDeviceMemory)
            {
                NVCudaWrapper.cuMemFree_v2(_pointer);
            }
        }
        #endregion
    }
}
