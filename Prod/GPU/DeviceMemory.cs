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
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering DeviceMemory(" + sizeInBytes + ")");}

            SizeInBytes = sizeInBytes;
            var res = NVCudaWrapper.cuMemAlloc_v2(out _pointer, SizeInBytes);
            GPUWrapper.CheckStatus(res);
            _isOwnerOfDeviceMemory = true;

            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving DeviceMemory(" + sizeInBytes + "), Pointer="+_pointer);}

        }
        /// <summary>
        /// Pointer to an already allocated memory on device.
        /// This memory area should not be freed when disposing the current object
        /// </summary>
        /// <param name="pointer">address of memory on device</param>
        /// <param name="sizeInBytes">size in bytes already allocated on device</param>
        public DeviceMemory(IntPtr pointer, size_t sizeInBytes)
        {
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering DeviceMemory(" + pointer+", "+sizeInBytes + ")");}

            SizeInBytes = sizeInBytes;
            _pointer = pointer;
            _isOwnerOfDeviceMemory = false;

            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving DeviceMemory(" + pointer + ", " + sizeInBytes + ")");}

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
                throw new Exception("disposed object of size " + SizeInBytes + " / _isOwnerOfDeviceMemory=" + _isOwnerOfDeviceMemory);
            }
        }

        public void ZeroMemory()
        {
            AssertIsNotDisposed();
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
            if (!_isOwnerOfDeviceMemory)
            {
                return;
            }


            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering ~DeviceMemory(" + SizeInBytes + "), Pointer=" + _pointer + ", Disposed=" + _disposed);}

            Dispose(false);

            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving ~DeviceMemory(" + SizeInBytes + "), Pointer=" + _pointer + ", Disposed=" + _disposed);}

        }




        // ReSharper disable once UnusedParameter.Local
        private void Dispose(bool disposing)
        {
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering DeviceMemory.Dispose(" + SizeInBytes + "), Pointer=" + _pointer+", Disposed="+_disposed);}

            if (_disposed)
            {

                if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving DeviceMemory.Dispose(" + SizeInBytes + "), Pointer=" + _pointer + ", Disposed=" + _disposed);}

                return;
            }
            _disposed = true;
            //unmanaged memory
            if (_isOwnerOfDeviceMemory)
            {
                var res = NVCudaWrapper.cuMemFree_v2(_pointer);
                GPUWrapper.CheckStatus(res);
                _pointer = IntPtr.Zero;

                if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving DeviceMemory.Dispose(" + SizeInBytes + "), Pointer=" + _pointer + ", Disposed=" + _disposed);}

            }


        }
        #endregion
    }
}
