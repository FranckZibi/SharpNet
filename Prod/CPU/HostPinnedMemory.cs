using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace SharpNet.CPU
{
    public class HostPinnedMemory<T> : IDisposable where T : struct
    {
        private T[] _pinnedHostMemory;
        /// <summary>
        /// pointer to the memory area located in the host (= CPU)
        /// </summary>
        private IntPtr _hostMemoryPointer;
        private GCHandle _handle;
        private bool _disposed;

        public int Capacity => _pinnedHostMemory.Length;
        public int Length { get; private set; }

        public HostPinnedMemory(T[] hostMemoryToBePinned)
        {
            InitializeAnPin(hostMemoryToBePinned);
        }
        public IntPtr Pointer => _hostMemoryPointer;

        private void InitializeAnPin(T[] hostMemoryToBePinned)
        {
            Debug.Assert(hostMemoryToBePinned != null);
            _pinnedHostMemory = hostMemoryToBePinned;
            _handle = GCHandle.Alloc(_pinnedHostMemory, GCHandleType.Pinned);
            _hostMemoryPointer = _handle.AddrOfPinnedObject();
            Length = Capacity;
        }

        public void Resize(int nbElements)
        {
            Debug.Assert(!_disposed);
            if (nbElements <= Capacity)
            {
                Length = nbElements;
                return;
            }
            if (_handle.IsAllocated)
            {
                _handle.Free();
            }
            InitializeAnPin(new T[nbElements]);
        }


        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
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
                GC.SuppressFinalize(this);
                if (_handle.IsAllocated)
                {
                    _handle.Free();
                }
            }
            _hostMemoryPointer = IntPtr.Zero;
            _pinnedHostMemory = null;
        }
        ~HostPinnedMemory()
        {
            Dispose(false);
        }
        #endregion


    }
}