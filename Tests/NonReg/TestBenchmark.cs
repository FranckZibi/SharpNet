using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using log4net;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.Datasets.CFM60;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Models;
using SharpNet.Networks;
using SharpNetTests.GPU;
// ReSharper disable AccessToDisposedClosure
// ReSharper disable AccessToModifiedClosure
#pragma warning disable 162

namespace SharpNetTests.NonReg
{
    [TestFixture]
    [SuppressMessage("ReSharper", "UnreachableCode")]
    [SuppressMessage("ReSharper", "ConditionIsAlwaysTrueOrFalse")]
    [SuppressMessage("ReSharper", "HeuristicUnreachableCode")]
    public class TestBenchmark
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(Network));


        static TestBenchmark()
        {
            Utils.ConfigureThreadLog4netProperties(NetworkConfig.DefaultWorkingDirectory, "SharpNet_Benchmark");
        }


        [Test, Explicit]
        public void TestGPUBenchmark_Memory()
        {
            //check RAM => GPU Copy perf
            var tmp_2GB = new float[500 * 1000000];

            var gpu = TestGPUTensor.GpuWrapper;
            Console.WriteLine(gpu.ToString());
            double maxSpeed = 0;
            for (int i = 1; i <= 3; ++i)
            {
                Console.WriteLine(Environment.NewLine + "Loop#" + i);
                var sw = Stopwatch.StartNew();
                var tensors = new GPUTensor<float>[1];
                for(int t=0;t<tensors.Length;++t)
                {
                    tensors[t] = new GPUTensor<float>(new[] { tmp_2GB.Length}, tmp_2GB, gpu);
                }
                Console.WriteLine(gpu.ToString());
                foreach (var t in tensors)
                {
                    t.Dispose();
                }
                var speed = (tensors.Length*((double)tensors[0].CapacityInBytes) / sw.Elapsed.TotalSeconds)/1e9;
                maxSpeed = Math.Max(speed, maxSpeed);
                Console.WriteLine("speed: " + speed + " GB/s");
            }

            System.IO.File.AppendAllText(Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultWorkingDirectory, "GPUBenchmark_Memory.csv"),
                DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                + "2GB Copy CPU=>GPU;"
                + gpu.DeviceName()+";"
#if DEBUG
                +"DEBUG;"
#else
                + "RELEASE;"
#endif
                + maxSpeed + ";"
                + Environment.NewLine
                );
        }


        [Test, Explicit]
        public void BenchmarkDataAugmentation()
        {
            const bool useMultiThreading = true;
            const bool useMultiGpu = true;
            const int targetHeight = 118;
            const int targetWidth = 100;
            var miniBatchSize = 300;
            // ReSharper disable once ConditionIsAlwaysTrueOrFalse
            if (useMultiGpu) { miniBatchSize *= GPUWrapper.GetDeviceCount(); }
            var p = EfficientNetSample.Cancel();
            p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_IMAGENET;
            p.Config.BatchSize = miniBatchSize;
            var database = new CancelDatabase();
            //TODO Test with selection of only matching size input in the training set
            using var dataset = database.ExtractDataSet(e => CancelDatabase.IsValidNonEmptyCancel(e.Cancel), ResizeStrategyEnum.BiggestCropInOriginalImageToKeepSameProportion);
            
            //dataAugmentationConfig.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10;
            var xMiniBatchShape = new []{miniBatchSize, 3, targetHeight, targetWidth};
            var yMiniBatchShape = dataset.YMiniBatch_Shape(miniBatchSize);
            var rand = new Random(0);
            var shuffledElementId = Enumerable.Range(0, dataset.Count).ToArray();
            Utils.Shuffle(shuffledElementId, rand);

            var xOriginalNotAugmentedMiniBatchCpu = new CpuTensor<float>(xMiniBatchShape);
            var xBufferMiniBatchCpu = new CpuTensor<float>(xMiniBatchShape);
            var xBufferForDataAugmentedMiniBatch = new CpuTensor<float>(xMiniBatchShape);
            var yBufferMiniBatchCpu = new CpuTensor<float>(yMiniBatchShape);
            var imageDataGenerator = new ImageDataGenerator(p.DA);
            
            yBufferMiniBatchCpu.ZeroMemory();
            var swLoad = new Stopwatch();
            var swDA = new Stopwatch();

            int count = 0;
            for (int firstElementId = 0; firstElementId <= (dataset.Count - miniBatchSize); firstElementId += miniBatchSize)
            {
                count += miniBatchSize;
                int MiniBatchIdxToElementId(int miniBatchIdx) => shuffledElementId[firstElementId + miniBatchIdx];
                swLoad.Start();

                if (useMultiThreading)
                {
                    Parallel.For(0, miniBatchSize, indexInBuffer => dataset.LoadAt(MiniBatchIdxToElementId(indexInBuffer), indexInBuffer, xOriginalNotAugmentedMiniBatchCpu, yBufferMiniBatchCpu,false));
                }
                else
                {
                    for (int indexInMiniBatch = 0; indexInMiniBatch < miniBatchSize; ++indexInMiniBatch)
                    {
                        dataset.LoadAt(MiniBatchIdxToElementId(indexInMiniBatch), indexInMiniBatch, xOriginalNotAugmentedMiniBatchCpu, yBufferMiniBatchCpu, false);
                    }
                }
                swLoad.Stop();
                swDA.Start();
                int MiniBatchIdxToCategoryIndex(int miniBatchIdx) => dataset.ElementIdToCategoryIndex(MiniBatchIdxToElementId(miniBatchIdx));
                Lazy<ImageStatistic> MiniBatchIdxToImageStatistic(int miniBatchIdx) => new Lazy<ImageStatistic>(() => dataset.ElementIdToImageStatistic(MiniBatchIdxToElementId(miniBatchIdx), targetHeight, targetWidth));
                if (useMultiThreading)
                {
                    Parallel.For(0, miniBatchSize, indexInMiniBatch => imageDataGenerator.DataAugmentationForMiniBatch(
                                                       indexInMiniBatch,
                                                       xOriginalNotAugmentedMiniBatchCpu,
                                                       xBufferMiniBatchCpu,
                                                       yBufferMiniBatchCpu,
                                                       MiniBatchIdxToCategoryIndex,
                                                       MiniBatchIdxToImageStatistic,
                                                       dataset.MeanAndVolatilityForEachChannel,
                                                       dataset.GetRandomForIndexInMiniBatch(indexInMiniBatch),
                                                       xBufferForDataAugmentedMiniBatch));
                }
                else
                {
                    for (int indexInMiniBatch = 0; indexInMiniBatch < miniBatchSize; ++indexInMiniBatch)
                    {
                        imageDataGenerator.DataAugmentationForMiniBatch(indexInMiniBatch, xOriginalNotAugmentedMiniBatchCpu, xBufferMiniBatchCpu, yBufferMiniBatchCpu, MiniBatchIdxToCategoryIndex, MiniBatchIdxToImageStatistic, dataset.MeanAndVolatilityForEachChannel, dataset.GetRandomForIndexInMiniBatch(indexInMiniBatch), xBufferForDataAugmentedMiniBatch);
                    }
                }

                //var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(147.02734f, 60.003986f), Tuple.Create(141.81636f, 51.15815f), Tuple.Create(130.15608f, 48.55502f) };
                //var xCpuChunkBytes = xOriginalNotAugmentedMiniBatchCpu /*xBufferMiniBatchCpu*/.Select((n, c, val) => (byte)((val * meanAndVolatilityOfEachChannel[c].Item2 + meanAndVolatilityOfEachChannel[c].Item1)));
                //for (int i = 0; i < 10; ++i)
                //{
                //    SharpNet.Pictures.PictureTools.SaveBitmap(xCpuChunkBytes, i, System.IO.Path.Combine(NetworkConfig.DefaultLogDirectory, "Train"), shuffledElementId[i].ToString("D5"), "");
                //}

                swDA.Stop();
            }
            var comment = "count=" + count.ToString("D4") + ",miniBatchSize=" + miniBatchSize.ToString("D4") + ", useMultiThreading=" + (useMultiThreading ? 1 : 0);
            comment += " ; load into memory took " + swLoad.ElapsedMilliseconds.ToString("D4") + " ms";
            comment += " ; data augmentation took " + swDA.ElapsedMilliseconds.ToString("D4") + " ms";
            Log.Info(comment);
        }

        [Test, Explicit]
        public void BenchmarkLoadAt()
        {
            const bool useMultiThreading = true;
            const int miniBatchSize = 1024;
            var p = Cfm60NetworkSample.Default();
            p.Config.BatchSize = miniBatchSize;

            using var cfm60TrainingAndTestDataSet = new CFM60TrainingAndTestDataSet(p, s => AbstractModel.Log.Info(s));
            var dataset = (CFM60DataSet)cfm60TrainingAndTestDataSet.Training;

            var xMiniBatchShape = new[] { miniBatchSize, 3, dataset.Sample.Encoder_TimeSteps, p.CFM60HyperParameters.Encoder_InputSize };

            var yMiniBatchShape = dataset.YMiniBatch_Shape(miniBatchSize);
            var rand = new Random(0);
            var shuffledElementId = Enumerable.Range(0, dataset.Count).ToArray();
            Utils.Shuffle(shuffledElementId, rand);

            var xOriginalNotAugmentedMiniBatchCpu = new CpuTensor<float>(xMiniBatchShape);
            var yBufferMiniBatchCpu = new CpuTensor<float>(yMiniBatchShape);

            yBufferMiniBatchCpu.ZeroMemory();
            var swLoad = new Stopwatch();

            int count = 0;
            for (int firstElementId = 0; firstElementId <= (dataset.Count - miniBatchSize); firstElementId += miniBatchSize)
            {
                count += miniBatchSize;
                int MiniBatchIdxToElementId(int miniBatchIdx) => shuffledElementId[firstElementId + miniBatchIdx];
                swLoad.Start();
                if (useMultiThreading)
                {
                    Parallel.For(0, miniBatchSize, indexInBuffer => dataset.LoadAt(MiniBatchIdxToElementId(indexInBuffer), indexInBuffer, xOriginalNotAugmentedMiniBatchCpu, yBufferMiniBatchCpu, false));
                }
                else
                {
                    for (int indexInMiniBatch = 0; indexInMiniBatch < miniBatchSize; ++indexInMiniBatch)
                    {
                        dataset.LoadAt(MiniBatchIdxToElementId(indexInMiniBatch), indexInMiniBatch, xOriginalNotAugmentedMiniBatchCpu, yBufferMiniBatchCpu, false);
                    }
                }
                swLoad.Stop();
            }
            var comment = "count=" + count.ToString("D4") + ",miniBatchSize=" + miniBatchSize.ToString("D4") + ", useMultiThreading=" + (useMultiThreading ? 1 : 0);
            comment += " ; load into memory took " + swLoad.ElapsedMilliseconds.ToString("D4") + " ms";
            Log.Info(comment);
        }


        //gpu=>gpu (same device)
        [TestCase("gpu0", "gpu0"), Explicit]
        //gpu=>gpu (different device)
        [TestCase("gpu0", "gpu1")]
        //gpu=>cpu
        [TestCase("gpu0", "cpu")]
        //cpu=>gpu
        [TestCase("cpu", "gpu0")]
        public void Test_MemoryCopy_Benchmark(string srcDescription, string destDescription)
        {
            var chunkSize = new[] { 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000 };
            var maxChunkSize = chunkSize.Max();
            var src = GetTensor(srcDescription, maxChunkSize);
            var dest = GetTensor(destDescription, maxChunkSize);
            foreach (var byteCount in chunkSize)
            { 
                ulong loopId = 0;
                src.Reshape(new[] { byteCount });
                dest.Reshape(new[] { byteCount });
                var sw = Stopwatch.StartNew();
                while (sw.ElapsedMilliseconds < 5000)
                {
                    src.CopyTo(dest);
                    ++loopId;
                }
                sw.Stop();
                var speed = (loopId* src.ReallyNeededMemoryInBytes / sw.Elapsed.TotalSeconds) / 1e9;
                Console.WriteLine("ByteCount: "+Utils.MemoryBytesToString((ulong)byteCount) + ", Avg speed: " + speed + " GB/s");
                System.IO.File.AppendAllText(Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultWorkingDirectory, "MemoryCopy_Benchmark.csv"), DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"+ srcDescription + ";"+ destDescription + ";"+ byteCount + ";"+ speed + ";"+ Environment.NewLine);
            }
        }
        private static Tensor GetTensor(string tensorDescription, int chunkSize)
        {
            if (tensorDescription == "gpu1" && GPUWrapper.GetDeviceCount() < 2)
            {
                tensorDescription = "gpu0";
            }
            switch (tensorDescription)
            {
                default:
                    //case "gpu":
                    //case "gpu0":
                    return new GPUTensor<byte>(new[] { chunkSize }, null, GPUWrapper.FromDeviceId(0));
                case "gpu1":
                    return new GPUTensor<byte>(new[] { chunkSize }, null, GPUWrapper.FromDeviceId(1));
                case "cpu":
                    return new CpuTensor<byte>(new[] { chunkSize }, null);
            }
        }
        [Test, Explicit]
        public void TestGPUBenchmark_Speed()
        {
            var mnist = new MNISTDataSet();
            const double learningRate = 0.01;
            var network = new Network(
                new NetworkConfig()
                {
                    ModelName = "GPUBenchmark",
                    ResourceIds = new List<int> { 0 },
                    BatchSize = 64,
                    NumEpochs = 5,
                    DisableReduceLROnPlateau = true
                }.WithAdam()
                .WithConstantLearningRateScheduler(learningRate)
                , 
                new DataAugmentationSample()
                );
            network
                .Input(MNISTDataSet.Shape_CHW)
                .Convolution(16, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .BatchNorm(0.99, 1e-5)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)
                .MaxPooling(2, 2, 2, 2)

                .Convolution(32, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)

                .Dense_Activation(1000, 0.0, false, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)

                .Output(MNISTDataSet.CategoryCount, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var sw = Stopwatch.StartNew();
            network.Fit(mnist.Training, mnist.Test);
            var elapsedMs = sw.Elapsed.TotalSeconds;
            var lossAndAccuracy = network.ComputeMetricsForTestDataSet(network.Config.BatchSize, mnist.Test);

            System.IO.File.AppendAllText(Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultWorkingDirectory, "GPUBenchmark_Speed.csv" ), 
                DateTime.Now.ToString("F", CultureInfo.InvariantCulture) +";"
                +"MNIST;"
                + network.DeviceName() + ";"
                + network.TotalParams() + ";"
                + network.Config.NumEpochs + ";"
                + network.Config.BatchSize + ";"
                + learningRate + ";"
#if DEBUG
                +"DEBUG;"
#else
                + "RELEASE;"
#endif
                +elapsedMs+";"
                +lossAndAccuracy[MetricEnum.Loss]+";"
                +lossAndAccuracy[MetricEnum.Accuracy]
                +Environment.NewLine
                );
        }
    }
}
