
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

        public HostPinnedMemory(Memory<T> hostMemoryToBePinned)
        {
            _handle = hostMemoryToBePinned.Pin();
            Pointer = (IntPtr)_handle.Pointer;
        }
        public IntPtr Pointer { get; private set; }

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
                _handle.Dispose();
            }
            //unmanaged memory
            Pointer = IntPtr.Zero;
        }
        ~HostPinnedMemory()
        {
            Dispose(false);
        }
        #endregion
   }
}
