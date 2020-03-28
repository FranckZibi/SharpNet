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

namespace SharpNet.Data
{
    public static unsafe class HDFUtils
    {
        /// <summary>
        /// Paths of all objects in the group
        /// </summary>
        /// <param name="groupId"></param>
        /// <returns></returns>
        public static List<string> ObjectPaths(H5GroupId groupId)
        {
            var objectPaths = new List<string>();
            var groupNamesHandle = GCHandle.Alloc(objectPaths);
            ulong pos = 0;
            H5L.iterate(groupId, H5.index_t.NAME, H5.iter_order_t.NATIVE, ref pos, SubGroupNamesDelegateMethod, (IntPtr)groupNamesHandle);
            groupNamesHandle.Free();
            return objectPaths;
        }
        public static H5O.type_t GetObjectType(H5GroupId groupId, string path)
        {
            H5O.info_t objectInfo = new H5O.info_t();
            int result = H5O.get_info_by_name(groupId, path, ref objectInfo);
            if (result < 0)
            {
                return H5O.type_t.UNKNOWN;
            }
            return objectInfo.type;
        }
        public static CpuTensor<T> LoadTensor<T>(H5ObjectId objectId, string datasetPath) where T : struct
        {
            var datasetId = H5D.open(objectId, datasetPath);
            var shape = DatasetShape(datasetId);

            var dataType = H5D.get_type(datasetId);
            if (H5T.get_class(dataType) == H5T.class_t.STRING)
            {
                var typeSize = H5T.get_size(dataType).ToInt32();
                shape = shape.Append(typeSize).ToArray();
            }

            long entireDataSetLength = Utils.Product(shape);
            var result = new T[entireDataSetLength];
            var resultHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            H5D.read(datasetId, dataType, H5S.ALL, H5S.ALL, H5P.DEFAULT, resultHandle.AddrOfPinnedObject());
            resultHandle.Free();
            H5D.close(datasetId);
            return new CpuTensor<T>(shape, result, datasetPath);
        }
        public static List<string> LoadLabelDataset(H5ObjectId objectId, string datasetPath)
        {
            var byteTensor = LoadTensor<byte>(objectId, datasetPath);
            Debug.Assert(byteTensor.Shape.Length == 2);
            var result = new List<string>();
            var stringMaxLength = byteTensor.Shape[1];
            for (int stringIndex = 0; stringIndex < byteTensor.Shape[0];++stringIndex)
            {
                result.Add(ExtractString(byteTensor.Content, stringIndex, stringMaxLength));
            }
            return result;
        }
        public static bool IsStringDataSet(H5DatasetId datasetId)
        {
            return GetClass(datasetId) == H5T.class_t.STRING;
        }

        public static string Join(string path1, string path2)
        {
            path1 = path1.TrimEnd('/');
            path2 = path2.TrimStart('/');

            if (string.IsNullOrEmpty(path1))
            {
                return path2;
            }
            if (string.IsNullOrEmpty(path2))
            {
                return path1;
            }
            return path1 + "/" + path2;
        }
        public static string ExtractString(byte[] dDim, int stringIndex, int stringMaxLength)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < stringMaxLength; ++i)
            {
                var c = (char)dDim[stringIndex * stringMaxLength + i];
                if (c == 0)
                {
                    break;
                }
                sb.Append(c);
            }
            return sb.ToString();
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
        private static H5T.class_t GetClass(H5DatasetId datasetId)
        {
            var dataType = H5D.get_type(datasetId);
            return H5T.get_class(dataType);
        }

    }
}