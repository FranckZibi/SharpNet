using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using JetBrains.Annotations;
using SharpNet.Data;

namespace SharpNet.GPU
{
    //This below code was inspired by ManagedCuda (http://kunzmi.github.io/managedCuda)
    public class CudaRuntimeCompiler : IDisposable
	{
        private readonly NVRTCWrapper _nvrtcWrapper;
	    private IntPtr _programHandle;
		private bool _disposed;
        public byte[] FatBinaryObject { get; private set; }


		public CudaRuntimeCompiler(string src, string name, Version cudaVersion)
        {
            _nvrtcWrapper =  new NVRTCWrapper(cudaVersion);
			var res = _nvrtcWrapper.nvrtcCreateProgram(out _programHandle, src, name, 0, null, null);
            GPUWrapper.CheckStatus(res);
	    }
	    public void Compile([NotNull] string[] options)
        {
            var optionsPtr = options.Select(Marshal.StringToHGlobalAnsi).ToArray();
            nvrtcResult res;
            try
            {
                res = _nvrtcWrapper.nvrtcCompileProgram(_programHandle, options.Length, optionsPtr);
            }
            finally
            {
                Array.ForEach(optionsPtr, Marshal.FreeHGlobal);
            }

            GPUWrapper.CheckStatus(res);
            res = _nvrtcWrapper.nvrtcGetPTXSize(_programHandle, out var ptxSize);
	        GPUWrapper.CheckStatus(res);
	        FatBinaryObject = new byte[ptxSize];
	        res = _nvrtcWrapper.nvrtcGetPTX(_programHandle, FatBinaryObject);
	        GPUWrapper.CheckStatus(res);
	    }
	    public string GetLogAsString()
	    {
	        var res = _nvrtcWrapper.nvrtcGetProgramLogSize(_programHandle, out size_t logSize);
	        GPUWrapper.CheckStatus(res);
	        byte[] logCode = new byte[logSize];
	        res = _nvrtcWrapper.nvrtcGetProgramLog(_programHandle, logCode);
	        GPUWrapper.CheckStatus(res);
            var enc = new ASCIIEncoding();
	        string logString = enc.GetString(logCode);
	        return logString.Replace("\0", "");
	    }

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
				_nvrtcWrapper.nvrtcDestroyProgram(ref _programHandle);
            }
        }
	    ~CudaRuntimeCompiler()
	    {
	        Dispose(false);
	    }
        #endregion
    }
}
