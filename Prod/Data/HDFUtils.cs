using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using HDF.PInvoke;
using SharpNet.CPU;

using H5GroupId = System.Int64;
using H5DatasetId = System.Int64;
using H5ObjectId = System.Int64;
using H5FileId = System.Int64;
using H5TypeId = System.Int64;

namespace SharpNet.Data
{
    public static unsafe class HDFUtils
    {
        public static Tensor SingleDataset(H5ObjectId objectId, string datasetPath)
        {
            var datasetId = H5D.open(objectId, datasetPath);
            var shape = DatasetShape(datasetId);
            var type = H5D.get_type(datasetId);
            H5D.close(datasetId);
            H5T.class_t typeClass = H5T.get_class(type);
            var typeSize = H5T.get_size(type).ToInt32();

            switch (typeClass)
            {
                case H5T.class_t.STRING:
                    shape = shape.Append(typeSize).ToArray();
                    return ToStringListTensor(LoadDataSet<byte>(objectId, datasetPath, shape));
                case H5T.class_t.FLOAT:
                    if (typeSize == 4)
                    {
                        return LoadDataSet<float>(objectId, datasetPath, shape);
                    }
                    if (typeSize == 8)
                    {
                        return LoadDataSet<double>(objectId, datasetPath, shape);
                    }
                    break;
                case H5T.class_t.INTEGER:
                    if (typeSize == 1)
                    {
                        return LoadDataSet<byte>(objectId, datasetPath, shape);
                    }
                    if (typeSize == 4)
                    {
                        return LoadDataSet<int>(objectId, datasetPath, shape);
                    }
                    if (typeSize == 8)
                    {
                        return LoadDataSet<long>(objectId, datasetPath, shape);
                    }
                    break;
            }
            throw new NotImplementedException("can't load tensor of type:" + type + "/typeClass:" + typeClass+"/typeSize:" + typeSize + " for dataset path " + datasetPath);
        }
        /// <summary>
        /// split a path to a dataSet between the path and the dataSetName
        /// </summary>
        /// <param name="datasetPath">the datasetPath to split (ex: /batch_normalization_10/batch_normalization_10/beta:0)</param>
        /// <param name="path">the associate path (ex: /batch_normalization_10/batch_normalization_10/)</param>
        /// <param name="datasetName">the name of the dataset (ex: beta:0)</param>
        public static void SplitPathAndName(string datasetPath, out string path, out string datasetName)
        {
            int lastIndex = datasetPath.LastIndexOf('/');
            if (lastIndex < 0)
            {
                path = "";
                datasetName = datasetPath;
            }
            else
            {
                path = datasetPath.Substring(0, lastIndex + 1);
                datasetName = datasetPath.Substring(lastIndex + 1);
            }
        }
        public static string Join(string path1, string path2)
        {
            if (string.IsNullOrEmpty(path2))
            {
                return path1;
            }
            if (path1.EndsWith("/") && path2.StartsWith("/"))
            {
                return path1+path2.TrimStart('/');
            }
            if (!path1.EndsWith("/") && !path2.StartsWith("/"))
            {
                return path1 + "/" + path2;
            }
            return path1 + path2;
        }
        /// <summary>
        /// extract the string at index 'stringIndex' from the buffer 'byteBuffer'
        /// the byteBuffer contains exactly 'byteBuffer.Length/stringMaxLength' strings
        ///  the first string starts at byteBuffer[0]
        ///  the 2nd string starts at byteBuffer[1*stringMaxLength]
        ///  etc...
        /// this method is public for testing only
        /// </summary>
        /// <param name="byteBuffer"></param>
        /// <param name="stringIndex"></param>
        /// <param name="stringMaxLength">max length of each string in the buffer</param>
        /// <returns></returns>
        public static string ExtractString(ReadOnlySpan<byte> byteBuffer, int stringIndex, int stringMaxLength)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < stringMaxLength; ++i)
            {
                var c = (char)byteBuffer[stringIndex * stringMaxLength + i];
                if (c == 0)
                {
                    break;
                }
                sb.Append(c);
            }
            return sb.ToString();
        }
        /// <summary>
        /// add the dataset 'dataSet' in the file 'fileId' at path 'datasetPathWithName'
        /// </summary>
        /// <typeparam name="T">the type of data to store (ex: float int, etc...)</typeparam>
        /// <param name="fileId">the file where to store the dataSet</param>
        /// <param name="datasetPathWithName">the path where to store the dataSet. It will be automatically created if needed</param>
        /// <param name="dataSet">the dataSet to store in the file</param>
        /// <returns>true if successful; otherwise false</returns>
        public static bool WriteDataSet<T>(H5FileId fileId, string datasetPathWithName, CpuTensor<T> dataSet)
        {
            SplitPathAndName(datasetPathWithName, out string path, out string name);
            H5GroupId groupId = OpenOrCreateGroupPath(fileId, path);
            var dims = dataSet.Shape.Select(h => (ulong)h).ToArray();
            var dsp = H5S.create_simple(dims.Length, dims, null);
            H5TypeId typeId = ToH5TypeId(typeof(T));
            var datasetId = H5D.create(groupId, name, typeId, dsp);
            var result = H5D.write(datasetId, H5T.NATIVE_FLOAT, H5S.ALL, H5S.ALL, H5P.DEFAULT, dataSet.Pointer);
            H5D.flush(datasetId);
            H5D.close(datasetId);
            H5S.close(dsp);
            H5G.close(groupId);
            return result>=0;
        }

        /// <summary>
        /// list of all dataset objects paths contained in a specific group.
        /// only the names (paths) of the dataset are returned (not the dataset content)
        /// </summary>
        /// <param name="fileId"></param>
        /// <param name="groupPath">group where to look for datasets</param>
        /// <returns>list of all dataset paths contained in the group</returns>
        public static List<string> DatasetPaths(H5FileId fileId, string groupPath = "/")
        {
            H5GroupId groupId = H5G.open(fileId, groupPath);
            var result = new List<string>();
            foreach (var objectName in ObjectNames(groupId))
            {
                var objectPath = Join(groupPath, objectName);
                var objectType = GetObjectType(groupId, objectName);
                if (objectType == H5O.type_t.DATASET)
                {
                    result.Add(objectPath);
                }
                else if (objectType == H5O.type_t.GROUP)
                {
                    result.AddRange(DatasetPaths(fileId, objectPath));
                }
                else
                {
                    throw new NotImplementedException("can't process object type " + objectType + " for path " + objectPath);
                }
            }
            H5G.close(groupId);
            return result;
        }
        public static H5TypeId ToH5TypeId(Type t)
        {
            if (t == typeof(float))
            {
                return H5T.NATIVE_FLOAT;
            }
            throw new NotImplementedException("type "+t+" is not supported");
        }

        /// <summary>
        /// name of all objects in the group
        /// </summary>
        /// <param name="groupId"></param>
        /// <returns></returns>
        private static List<string> ObjectNames(H5GroupId groupId)
        {
            var objectPaths = new List<string>();
            GCHandle groupNamesHandle = GCHandle.Alloc(objectPaths);
            ulong pos = 0;
            H5L.iterate(groupId, H5.index_t.NAME, H5.iter_order_t.NATIVE, ref pos, SubGroupNamesDelegateMethod, (IntPtr)groupNamesHandle);
            groupNamesHandle.Free();
            return objectPaths;
        }
        private static int SubGroupNamesDelegateMethod(H5GroupId group, IntPtr name, ref H5L.info_t info, IntPtr op_data)
        {
            GCHandle hnd = (GCHandle)op_data;
            var list = (List<string>)hnd.Target;
            int len = 0;
            while (Marshal.ReadByte(name, len) != 0)
            {
                ++len;
            }
            var nameBuffer = new byte[len];
            Marshal.Copy(name, nameBuffer, 0, len);
            list.Add(Encoding.UTF8.GetString(nameBuffer));
            return 0;
        }
        /// <summary>
        /// retrieves the shape of the dataset 'datasetId'
        /// </summary>
        /// <param name="datasetId"></param>
        /// <returns></returns>
        private static int[] DatasetShape(H5DatasetId datasetId)
        {
            var dataSpace = H5D.get_space(datasetId);
            var dimension = H5S.get_simple_extent_ndims(dataSpace);
            var datasetShape = new ulong[dimension];
            var datasetShapeHandle = GCHandle.Alloc(datasetShape, GCHandleType.Pinned);
            H5S.get_simple_extent_dims(dataSpace, (ulong*)datasetShapeHandle.AddrOfPinnedObject(), null);
            datasetShapeHandle.Free();
            H5S.close(dataSpace);
            return datasetShape.Select(x => (int)x).ToArray();
        }
        private static CpuTensor<T> LoadDataSet<T>(H5DatasetId objectId, string datasetPath, int[] datasetShape)
        {
            var datasetId = H5D.open(objectId, datasetPath);
            var dataType = H5D.get_type(datasetId);
            long entireDataSetLength = Utils.Product(datasetShape);
            var result = new T[entireDataSetLength];
            var resultHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            H5D.read(datasetId, dataType, H5S.ALL, H5S.ALL, H5P.DEFAULT, resultHandle.AddrOfPinnedObject());
            resultHandle.Free();
            H5D.close(datasetId);
            return new CpuTensor<T>(datasetShape, result);
        }
        private static H5O.type_t GetObjectType(H5GroupId groupId, string path)
        {
            H5O.info_t objectInfo = new H5O.info_t();
            int result = H5O.get_info_by_name(groupId, path, ref objectInfo);
            if (result < 0)
            {
                return H5O.type_t.UNKNOWN;
            }
            return objectInfo.type;
        }
        private static CpuTensor<string> ToStringListTensor(CpuTensor<byte> byteTensor)
        {
            Debug.Assert(byteTensor.Shape.Length == 2);
            var result = new string[byteTensor.Shape[0]];
            var stringMaxLength = byteTensor.Shape[1];
            for (int stringIndex = 0; stringIndex < byteTensor.Shape[0]; ++stringIndex)
            {
                result[stringIndex] = ExtractString(byteTensor.ReadonlyContent, stringIndex, stringMaxLength);
            }
            return new CpuTensor<string>(new[] { byteTensor.Shape[0] }, result, IntPtr.Size);
        }
        private static H5GroupId OpenOrCreateGroupPath(H5FileId fileId, string groupPath)
        {
            var allExistingDatasets = DatasetPaths(fileId, "/").Select(s => s.TrimStart('/')).ToList();
            var splitted = groupPath.Trim('/').Split('/').ToArray();
            for (int i = 1; i <= splitted.Length; ++i)
            {
                var subPart = string.Join("/", splitted.Take(i)) + "/";
                if (!allExistingDatasets.Any(t => t.StartsWith(subPart)))
                {
                    var groupId = H5G.create(fileId, subPart);
                    if (groupId < 0)
                    {
                        return -1;
                    }
                    H5G.close(groupId);
                }
            }
            return H5G.open(fileId, groupPath);
        }
    }
}