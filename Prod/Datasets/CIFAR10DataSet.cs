using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CPU;
using SharpNet.Networks;
/*
BatchSize = 128
EpochCount = 150
SGD with momentum = 0.9 & L2 = 0.5* 1-e4
CutMix / no Cutout / no Mixup / FillMode = Reflect / Disable DivideBy10OnPlateau
AvgPoolingStride = 2
# ------------------------------------------------------------------------------------------------
#           |             |    200-epoch  |   150-epoch   |   200-epoch   |   Orig Paper  | sec/epoch
# Model     |   #Params   |      WRN      |   SGDR 10-2   |   SGDR 10-2   |      WRN      | GTX1080
#           |             |   %Accuracy   |   %Accuracy   |   %Accuracy   |   %Accuracy   | 
#           |             |   (dropout)   |(Ens. Learning)|(Ens. Learning)|   (dropout)   | 
# -----------------------------------------------------------------------------------------------
# WRN-16-4  |   2,752,506 | 94.61 (-----) | 95.67 (--.--) | 94.99 (95.69) | 94.98 (94.76) |  27.4
# WRN-40-4  |   8,959,994 | 95.43 (-----) | 96.29 (96.18) | 95.67 (96.30) | 95.43 (-----) |  77.3
# WRN-16-8  |  10,968,570 | 95.20 (-----) | 95.99 (--.--) | 95.69 (96.03) | 95.73         |  83.0
# WRN-16-10 |  17,125,626 | ----- (-----) | ----- (-----) | ----- (-----) | NA            | 136.0
# WRN-28-8  |  23,369,210 | ----- (-----) | ----- (-----) | ----- (-----) | NA            | 173.0
# WRN-28-10 |  36,497,146 | 95.28 (-----) | 96.40 (96.79) | ----- (-----) | 96.00 (96.11) | 296.5
# -----------------------------------------------------------------------------------------------
*/

namespace SharpNet.Datasets
{
    public class CIFAR10DataSet : AbstractTrainingAndTestDataset
    {
        private static readonly string[] CategoryIndexToDescription = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };
        public static int CategoryCount => CategoryIndexToDescription.Length;

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }

        public static readonly int[] Shape_CHW = { 3, 32, 32 };

        public const string NAME = "CIFAR-10";

        public CIFAR10DataSet() : base(NAME)
        {
            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, Name);

            //We load the training set
            var xTrainingSet = new CpuTensor<byte>(new[] { 50000, Shape_CHW[0], Shape_CHW[1], Shape_CHW[2] });
            var yTrainingSet = new CpuTensor<byte>(new[] { 50000, 1, 1, 1 });
            for (int i = 0; i < 5; ++i)
            {
                LoaAllFileAt(Path.Combine(path, "data_batch_" + (i + 1) + ".bin"), xTrainingSet, yTrainingSet, 10000 * i);
            }
            //We normalize the input with 0 mean / 1 volatility
            //var meanAndVolatilityOfEachChannelInTrainingSet = xTrainingSet.ComputeMeanAndVolatilityOfEachChannel(x=>(double)x);
            var meanAndVolatilityOfEachChannelInTrainingSet = new List<Tuple<float, float>> { Tuple.Create(125.306918046875f, 62.9932192781369f), Tuple.Create(122.950394140625f, 62.0887076400142f), Tuple.Create(113.865383183594f, 66.7048996406309f) };
            var xTrain = AbstractDataSet.ToXWorkingSet(xTrainingSet, meanAndVolatilityOfEachChannelInTrainingSet);
            var yTrain = AbstractDataSet.ToYWorkingSet(yTrainingSet, CategoryCount, CategoryByteToCategoryIndex);

            AbstractDataSet.AreCompatible_X_Y(xTrain, yTrain);

            //We load the test set
            var xTestSet = new CpuTensor<byte>(new[] { 10000, Shape_CHW[0], Shape_CHW[1], Shape_CHW[2] });
            var yTestSet = new CpuTensor<byte>(new[] { 10000, 1, 1, 1 });
            LoaAllFileAt(Path.Combine(path, "test_batch.bin"), xTestSet, yTestSet, 0);
            //We normalize the test set with 0 mean / 1 volatility (coming from the training set)
            var xTest = AbstractDataSet.ToXWorkingSet(xTestSet, meanAndVolatilityOfEachChannelInTrainingSet);
            var yTest = AbstractDataSet.ToYWorkingSet(yTestSet, CategoryCount, CategoryByteToCategoryIndex);

            AbstractDataSet.AreCompatible_X_Y(xTest, yTest);

            //Uncomment the following line to take only the first 'count' elements
            //const int count = 1000;xTrain = (CpuTensor<float>)xTrain.RowSlice(0, count);yTrain = (CpuTensor<float>)yTrain.RowSlice(0, xTrain.Shape[0]); xTest = (CpuTensor<float>)xTest.RowSlice(0, count); ; yTest = (CpuTensor<float>)yTest.RowSlice(0, xTest.Shape[0]);

            Training = new InMemoryDataSet(xTrain, yTrain, Name, Objective_enum.Classification, meanAndVolatilityOfEachChannelInTrainingSet, CategoryIndexToDescription);
            Test = new InMemoryDataSet(xTest, yTest, Name, Objective_enum.Classification, meanAndVolatilityOfEachChannelInTrainingSet, CategoryIndexToDescription);
        }

        private static void LoaAllFileAt(string path, CpuTensor<byte> x, CpuTensor<byte> categoryBytes, int indexFirst)
        {
            Debug.Assert(x.Shape[1] == Shape_CHW[0]); //channels
            Debug.Assert(x.Shape[2] == Shape_CHW[1]); //height
            Debug.Assert(x.Shape[3] == Shape_CHW[2]); //width
            int bytesInSingleElement = x.MultDim0 + 1;

            int elementCountInPath = (int)(Utils.FileLength(path) / bytesInSingleElement);
            var b = File.ReadAllBytes(path);
            for (int count = 0; count < elementCountInPath; ++count)
            {
                int bIndex = count * bytesInSingleElement;
                int xIndex = (count + indexFirst) * x.MultDim0;
                int yIndex = (count + indexFirst) * categoryBytes.MultDim0;
                categoryBytes[yIndex] = b[bIndex];
                for (int j = 0; j < (bytesInSingleElement - 1); ++j)
                {
                    x[xIndex + j] = b[bIndex + 1 + j];
                }
            }
        }

    }

}
