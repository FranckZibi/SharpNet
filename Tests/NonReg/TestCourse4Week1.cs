using System.Diagnostics;
using NUnit.Framework;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using HDF5DotNet;
using SharpNet;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Pictures;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestCourse4Week1
    {
        [Test, Explicit]
        [SuppressMessage("ReSharper", "ConditionIsAlwaysTrueOrFalse")]
        public void Test()
        {
            LoadDataSet(@"C:\Projects\SharpNet\Tests\Data\Course4Week1\train_signs.h5", "train_set_x", "train_set_y",
                "list_classes", out var train_set_x_orig, out var train_set_y_orig, out _);
            var train_set_x = train_set_x_orig.From_NHWC_to_NCHW(x => x / 255.0);
            var train_set_y = train_set_y_orig.ToCategorical(1.0, out _);

            LoadDataSet(@"C:\Projects\SharpNet\Tests\Data\Course4Week1\test_signs.h5", "test_set_x", "test_set_y",
                "list_classes", out var test_set_x_orig, out var test_set_y_orig, out _);
            var test_set_x = test_set_x_orig.From_NHWC_to_NCHW(x => x / 255.0);
            var test_set_y = test_set_y_orig.ToCategorical(1.0, out _);

            var logger = new Logger(LogFileName, true);

            var gpuDeviceId = 0;
            var network = new Network(new NetworkConfig() {Logger = logger, UseDoublePrecision = false}.WithAdam(), ImageDataGenerator.NoDataAugmentation, gpuDeviceId);
            var relu = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;
            double lambdaL2Regularization = 0.0;
            network
                .Input(train_set_x.Shape[1], train_set_x.Shape[2], train_set_x.Shape[3])
                .Convolution_Activation_Pooling(8, 3, 1, 1, lambdaL2Regularization, relu, 2, 2)
                .Convolution_Activation_Pooling(16, 3, 1, 1, lambdaL2Regularization, relu, 2, 2)
                .Convolution_Activation_Pooling(32, 3, 1, 1, lambdaL2Regularization, relu, 2, 2)
                .Dense_Activation(100, 0, relu)
                .Output(train_set_y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
            network.Fit(train_set_x, train_set_y, 0.009, 100, 10, test_set_x, test_set_y);
        }

        private static string LogFileName => Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory,
            "Course1Week4" + "_" + Process.GetCurrentProcess().Id + "_" +
            System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");

        private static void LoadDataSet(string hdf5FileName, string featuresDataSetName, string labelsDataSetName,
            string labelListName, out CpuTensor<byte> features, out CpuTensor<double> labels,
            out List<string> labelNames)
        {
            H5.Open();
            //Console.WriteLine("HDF5 " + H5.Version.Major + "." + H5.Version.Minor + "." + H5.Version.Release);
            var fileID = H5F.open(hdf5FileName, H5F.OpenMode.ACC_RDONLY);
            var groupId = H5G.open(fileID, "/");

            if (GetClass(groupId, labelListName) == H5T.H5TClass.STRING)
            {
                labelNames = LoadStrings(groupId, labelListName);
            }
            else
            {
                labelNames = LoadContent<long>(groupId, labelListName, out _, out _).Select(x => x.ToString()).ToList();
            }

            features = LoadFeaturesDataSet(groupId, featuresDataSetName);
            labels = LoadLabelsDataSet(groupId, labelsDataSetName);
            Debug.Assert(features.Height == labels.Count);
        }

        private static CpuTensor<byte> LoadFeaturesDataSet(H5GroupId groupId, string dataSetName)
        {
            int[] size;
            var dDim = LoadContent<byte>(groupId, dataSetName, out size, out _);
            Debug.Assert(size.Length == 4);
            return new CpuTensor<byte>(size, dDim, dataSetName);
        }

        private static CpuTensor<double> LoadLabelsDataSet(H5GroupId groupId, string dataSetName)
        {
            var content = LoadContent<long>(groupId, dataSetName).Select(x => (double) x).ToArray();
            return new CpuTensor<double>(new[] {content.Length, 1}, content, dataSetName);
        }

        private static T[] LoadContent<T>(H5GroupId groupId, string dataSetName)
        {
            return LoadContent<T>(groupId, dataSetName, out _, out _);
        }

        private static H5T.H5TClass GetClass(H5GroupId groupId, string dataSetName)
        {
            var datasetID = H5D.open(groupId, dataSetName);
            var dataType = H5D.getType(datasetID);
            return H5T.getClass(dataType);
        }

        private static T[] LoadContent<T>(H5GroupId groupId, string dataSetName, out int[] size, out int typeSize)
        {
            var datasetID = H5D.open(groupId, dataSetName);
            var dataSpace = H5D.getSpace(datasetID);
            size = H5S.getSimpleExtentDims(dataSpace).Select(x => (int) x).ToArray();
            var dataType = H5D.getType(datasetID);
            H5T.H5TClass tcls = H5T.getClass(dataType);

            typeSize = H5T.getSize(dataType);
            long entireDataSetLength = Utils.Product(size);
            if (tcls == H5T.H5TClass.STRING)
            {
                entireDataSetLength *= typeSize;
            }
            var dDim = new T[entireDataSetLength];
            H5D.read(datasetID, dataType, new H5Array<T>(dDim));
            return dDim;
        }

        private static List<string> LoadStrings(H5GroupId groupId, string dataSetName)
        {
            int[] size;
            int typeSize;
            var dDim = LoadContent<byte>(groupId, dataSetName, out size, out typeSize);
            var result = new List<string>();
            while (result.Count < size[0])
            {
                result.Add(ExtractString(dDim, result.Count, typeSize));
            }
            return result;
        }

        private static string ExtractString(byte[] dDim, int stringIndex, int stringMaxLength)
        {
            var chars = new List<char>();
            for (int i = 0; i < stringMaxLength; ++i)
            {
                var c = (char) dDim[stringIndex * stringMaxLength + i];
                if (c == 0)
                {
                    break;
                }
                chars.Add(c);
            }
            return new string(chars.ToArray());
        }
    }
}
