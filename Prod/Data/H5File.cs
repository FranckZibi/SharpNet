using System;
using System.Collections.Generic;
using System.IO;
using HDF.PInvoke;

using H5GroupId = System.Int64;
using H5DatasetId = System.Int64;
using H5ObjectId = System.Int64;

namespace SharpNet.Data
{
    public class H5File : IDisposable
    {
        #region private fields
        private readonly string _filePath;
        private Int64 _fileId;
        private bool _disposed;
        #endregion

        public H5File(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new ArgumentException("missing file "+filePath);
            }

            _filePath = filePath;
            _fileId = H5F.open(filePath, H5F.ACC_RDONLY);
        }

        /// <summary>
        /// return all dataSets contained in the current file starting for 'groupPath'
        /// </summary>
        /// <returns>
        /// A list of dataset. for each element of the list:
        ///     Item1: full path to the data set
        ///     Item2: shape of the data set
        /// </returns>
        public List<string> DatasetPaths(string groupPath = "/")
        {
            H5GroupId groupId = H5G.open(_fileId, groupPath);
            var result = new List<string>();
            foreach (var objectName in HDFUtils.ObjectPaths(groupId))
            {
                var objectType = HDFUtils.GetObjectType(groupId, objectName);
                var objectPath = HDFUtils.Join(groupPath, objectName);
                if (objectType == H5O.type_t.DATASET)
                {
                    H5DatasetId datasetId = H5D.open(_fileId, objectPath);
                    result.Add(objectPath);
                    H5D.close(datasetId);
                }
                else if (objectType == H5O.type_t.GROUP)
                {
                    result.AddRange(DatasetPaths(objectPath));
                }
            }
            H5G.close(groupId);
            return result;
        }

        /// <summary>
        /// content of value dataset (dataset with float/double/byte/long/int, excluding string)
        /// </summary>
        /// <param name="groupPath"></param>
        /// <returns></returns>
        public List<Tuple<string, Tensor>> ValueDatasetsContent(string groupPath = "/")
        {
            H5GroupId groupId = H5G.open(_fileId, groupPath);
            var result = new List<Tuple<string, Tensor>>();
            foreach (var objectPath in DatasetPaths(groupPath))
            {
                H5DatasetId datasetId = H5D.open(_fileId, objectPath);
                if (!HDFUtils.IsStringDataSet(datasetId))
                {
                    Tensor tensor = HDFUtils.LoadTensor<float>(_fileId, objectPath);
                    result.Add(Tuple.Create(objectPath, tensor));
                }
                H5D.close(datasetId);
            }
            H5G.close(groupId);
            return result;
        }


        /// <summary>
        /// content of label dataset 
        /// </summary>
        /// <param name="groupPath"></param>
        /// <returns></returns>
        public List<Tuple<string, List<string>>> LabelDatasetsContent(string groupPath = "/")
        {
            H5GroupId groupId = H5G.open(_fileId, groupPath);
            var result = new List<Tuple<string, List<string>>>();
            foreach (var datasetPath in DatasetPaths(groupPath))
            {
                H5DatasetId datasetId = H5D.open(_fileId, datasetPath);
                if (HDFUtils.IsStringDataSet(datasetId))
                {
                    var labelNames = HDFUtils.LoadLabelDataset(groupId, datasetPath);
                    result.Add(Tuple.Create(datasetPath, labelNames));
                }
                H5D.close(datasetId);
            }
            H5G.close(groupId);
            return result;
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
                _fileId = 0;
            }
        }
        ~H5File()
        {
            Dispose(false);
        }
        #endregion

    }
}