using System;
using SharpNet.Data;

namespace SharpNet.GPU
{
    //This below code was inspired by ManagedCuda (http://kunzmi.github.io/managedCuda)
    public sealed class CudaDeviceMemory : IDisposable
    {
        #region Private fields
        private readonly IntPtr _devicePtr;
        private readonly size_t _sizeInBytes;
        private bool _disposed;
        #endregion
        public bool IsOwner { get; }

        public CudaDeviceMemory(size_t sizeInBytes)
        {
            _sizeInBytes = sizeInBytes;
            var res = NVCudaWrapper.cuMemAlloc_v2(out _devicePtr, _sizeInBytes);
            GPUWrapper.CheckStatus(res);
            IsOwner = true;
        }
        public CudaDeviceMemory(IntPtr devicePtr, bool isOwner)
        {
            _devicePtr = devicePtr;
            var res = NVCudaWrapper.cuMemGetAddressRange_v2(out _, out _sizeInBytes, devicePtr);
            GPUWrapper.CheckStatus(res);
            IsOwner = isOwner;
        }
        
        public IntPtr DevicePointer => _devicePtr;
        public size_t SizeInBytes => _sizeInBytes;
        public void ZeroMemory()
        {
            CUresult res;
            if (_sizeInBytes % 4 == 0)
            {
                res = NVCudaWrapper.cuMemsetD32_v2(_devicePtr, 0, _sizeInBytes / 4);
            }
            else
            {
                res = NVCudaWrapper.cuMemsetD8_v2(_devicePtr, (char)0, _sizeInBytes);
            }
            GPUWrapper.CheckStatus(res);
        }

        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        ~CudaDeviceMemory()
        {
            Dispose(false);
        }
        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            if (disposing && IsOwner)
            {
                NVCudaWrapper.cuMemFree_v2(_devicePtr);
            }
        }
        #endregion
    }
}
