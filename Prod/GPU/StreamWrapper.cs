using System;

namespace SharpNet.GPU
{
    public class StreamWrapper : IDisposable
    {
        #region private fields
        private bool _disposed;
        private readonly IntPtr _streamHandle;
        #endregion

        public StreamWrapper()
        {
            var res = NVCudaWrapper.cuStreamCreate(out _streamHandle, 0);
            GPUWrapper.CheckStatus(res);
        }
        public IntPtr StreamHandle => _streamHandle;
        public void Synchronize()
        {
            var res = NVCudaWrapper.cuStreamSynchronize(_streamHandle);
            GPUWrapper.CheckStatus(res);
        }

        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        ~StreamWrapper()
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
            if (disposing)
            {
                //managed memory
            }
            //unmanaged memory
            var cuRes = NVCudaWrapper.cuStreamDestroy_v2(_streamHandle);
            GPUWrapper.CheckStatus(cuRes);
        }
        #endregion
    }
}
