using System;
using System.Collections.Generic;
using System.IO;
using HDF.PInvoke;
using SharpNet.CPU;
using H5FileId = System.Int64;
using H5GroupId = System.Int64;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Data
{
    public class H5File : IDisposable
    {
        #region private fields
        private readonly string _filePath;
        private readonly H5FileId _fileId;
        private bool _disposed;
        #endregion

        public H5File(string filePath, uint flags = H5F.ACC_RDONLY)
        {
            _filePath = filePath;
            if (!File.Exists(filePath))
            {
                //we create the file
                _fileId = H5F.create(_filePath, H5F.ACC_TRUNC);
                if (_fileId < 0)
                {
                    throw new ApplicationException("H5F.create fail for file " + filePath);
                }
            }
            else
            {
                _fileId = H5F.open(_filePath, flags);
                if (_fileId < 0)
                {
                    throw new ApplicationException("H5F.open fail for file "+filePath);
                }
            }
        }

        /// <summary>
        /// all dataset objects in the provided group path
        /// </summary>
        /// <param name="groupPath">group where to look for datasets</param>
        /// <returns>the  list of all datasets contained in the group</returns>
        public IDictionary<string, Tensor> Datasets(string groupPath = "/")
        {
            H5GroupId groupId = H5G.open(_fileId, groupPath);
            var result = new Dictionary<string, Tensor>();
            foreach (var datasetPath in HDFUtils.DatasetPaths(_fileId, groupPath))
            {
                if (result.ContainsKey(datasetPath))
                {
                    throw new ArgumentException("duplicate dataset path for " + datasetPath);
                }
                result[datasetPath] = HDFUtils.SingleDataset(groupId, datasetPath);
            }
            H5G.close(groupId);
            return result;
        }

        /// <summary>
        /// add the dataset 'dataSet' in the file at path 'datasetPathWithName'
        /// </summary>
        /// <param name="datasetPathWithName">the path where to store the dataSet. It will be automatically created if needed</param>
        /// <param name="dataSet">the dataSet to store in the file</param>
        /// <returns>true if successful; otherwise false</returns>
        public bool Write<T>(string datasetPathWithName, CpuTensor<T> dataSet)
        {
            return HDFUtils.WriteDataSet(_fileId, datasetPathWithName, dataSet);
        }

        public override string ToString()
        {
            return _filePath;
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
                //managed memory
            }
            //unmanaged memory
            if (_fileId != 0)
            {
                H5F.close(_fileId);
            }
        }
        ~H5File()
        {
            Dispose(false);
        }
        #endregion
    }
}
