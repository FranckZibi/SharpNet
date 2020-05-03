using System;
using System.Buffers;

namespace SharpNet.CPU
{
    public unsafe class HostPinnedMemory<T> : IDisposable
    {
        #region private fields
        private MemoryHandle _handle;
        private bool _disposed;
        #endregion

        #region public properties
        public IntPtr Pointer { get; }
        #endregion

        public HostPinnedMemory(Memory<T> hostMemoryToBePinned)
        {
            _handle = hostMemoryToBePinned.Pin();
            Pointer = (IntPtr)_handle.Pointer;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _handle.Dispose();
        }
   }
}
