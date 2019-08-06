
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace SharpNet.CPU
{
    public class HostPinnedMemory<T> : IDisposable where T : struct
    {
        #region private fields
        private GCHandle _handle;
        private bool _disposed;
        #endregion

        public HostPinnedMemory(T[] hostMemoryToBePinned)
        {
            Debug.Assert(hostMemoryToBePinned != null);
            _handle = GCHandle.Alloc(hostMemoryToBePinned, GCHandleType.Pinned);
            Pointer = _handle.AddrOfPinnedObject();
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
                if (_handle.IsAllocated)
                {
                    _handle.Free();
                }
            }
            Pointer = IntPtr.Zero;
        }
        ~HostPinnedMemory()
        {
            Dispose(false);
        }
        #endregion
   }
}
