using System;
using System.Collections.Generic;
using System.IO;
using HDF.PInvoke;
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
        /// list of all dataset objects paths contained in a specific group.
        /// only the names (paths) of the dataset are returned (not the dataset content)
        /// </summary>
        /// <param name="groupPath">group where to look for datasets</param>
        /// <returns>list of all dataset paths contained in the group</returns>
        // ReSharper disable once MemberCanBePrivate.Global
        public List<string> DatasetPaths(string groupPath = "/")
        {
            H5GroupId groupId = H5G.open(_fileId, groupPath);
            var result = new List<string>();
            foreach (var objectName in HDFUtils.ObjectNames(groupId))
            {
                var objectPath = HDFUtils.Join(groupPath, objectName);
                var objectType = HDFUtils.GetObjectType(groupId, objectName);
                if (objectType == H5O.type_t.DATASET)
                {
                    result.Add(objectPath);
                }
                else if (objectType == H5O.type_t.GROUP)
                {
                    result.AddRange(DatasetPaths(objectPath));
                }
                else
                {
                    throw new NotImplementedException("can't process object type "+objectType+" for path "+objectPath+" in file "+_filePath );
                }
            }
            H5G.close(groupId);
            return result;
        }

        /// <summary>
        /// all dataset objects in the provided group path
        /// </summary>
        /// <param name="groupPath">group where to look for datasets</param>
        /// <returns>the  list of all datasets contained in the group</returns>
        public List<Tuple<string, Tensor>> Datasets(string groupPath = "/")
        {
            H5GroupId groupId = H5G.open(_fileId, groupPath);
            var result = new List<Tuple<string, Tensor>>();
            foreach (var datasetPath in DatasetPaths(groupPath))
            {
                var tensor = HDFUtils.SingleDataset(groupId, datasetPath);
                result.Add(Tuple.Create(datasetPath, tensor));
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
            }
        }
        ~H5File()
        {
            Dispose(false);
        }
        #endregion
    }
}
