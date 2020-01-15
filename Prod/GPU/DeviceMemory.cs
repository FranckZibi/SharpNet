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
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering DeviceMemory(" + sizeInBytes + ")");}
#endif
            SizeInBytes = sizeInBytes;
            var res = NVCudaWrapper.cuMemAlloc_v2(out _pointer, SizeInBytes);
            GPUWrapper.CheckStatus(res);
            _isOwnerOfDeviceMemory = true;
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving DeviceMemory(" + sizeInBytes + "), Pointer="+_pointer);}
#endif
        }
        /// <summary>
        /// Pointer to an already allocated memory on device.
        /// This memory area should not be freed when disposing the current object
        /// </summary>
        /// <param name="pointer">address of memory on device</param>
        /// <param name="sizeInBytes">size in bytes already allocated on device</param>
        public DeviceMemory(IntPtr pointer, size_t sizeInBytes)
        {
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering DeviceMemory(" + pointer+", "+sizeInBytes + ")");}
#endif
            SizeInBytes = sizeInBytes;
            _pointer = pointer;
            _isOwnerOfDeviceMemory = false;
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving DeviceMemory(" + pointer + ", " + sizeInBytes + ")");}
#endif
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
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering ~DeviceMemory(" + SizeInBytes + "), Pointer=" + _pointer + ", Disposed=" + _disposed);}
#endif
            Dispose(false);
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving ~DeviceMemory(" + SizeInBytes + "), Pointer=" + _pointer + ", Disposed=" + _disposed);}
#endif
        }




        // ReSharper disable once UnusedParameter.Local
        private void Dispose(bool disposing)
        {
#if DEBUG
            if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("entering DeviceMemory.Dispose(" + SizeInBytes + "), Pointer=" + _pointer+", Disposed="+_disposed);}
#endif
            if (_disposed)
            {
#if DEBUG
                if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving DeviceMemory.Dispose(" + SizeInBytes + "), Pointer=" + _pointer + ", Disposed=" + _disposed);}
#endif
                return;
            }
            _disposed = true;
            //unmanaged memory
            if (_isOwnerOfDeviceMemory)
            {
                var res = NVCudaWrapper.cuMemFree_v2(_pointer);
                //GPUWrapper.CheckStatus(res);
                _pointer = IntPtr.Zero;
#if DEBUG
                if (GPUWrapper.DEBUG_CUDA){GPUWrapper.LogDebug("leaving DeviceMemory.Dispose(" + SizeInBytes + "), Pointer=" + _pointer + ", Disposed=" + _disposed);}
#endif
            }


        }
#endregion
    }
}
