using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Layers;
using SharpNet.LightGBM;
using SharpNet.Networks;
using static SharpNet.Datasets.Natixis70.Natixis70Utils;

// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global

namespace SharpNet.Datasets.Natixis70
{
    public class Natixis70DataSet : InMemoryDataSet
    {
        public readonly Natixis70HyperParameters HyperParameters;


       
        private static (CpuTensor<float>, CpuTensor<float>) LoadX_Y(string xRawFile, string yRawFileIfAny, Natixis70HyperParameters hyperParameters)
        {
            //We load 'xRaw'
            var xRawDataframe = Dataframe.Load(xRawFile, true, ',');
            var xRaw = xRawDataframe.Tensor;
            Debug.Assert(xRaw.Shape[1] == EMBEDDING_DIMENSION);
            int count = xRaw.Shape[0];
            var xRawSpan = xRaw.AsReadonlyFloatCpuContent;
            var x = new CpuTensor<float>(hyperParameters.X_Shape(count));
            var xSpan = x.AsFloatCpuSpan;
            int xSpanIndex = 0;

            //We load 'yRaw' if needed
            CpuTensor<float> yRaw = null;
            ReadOnlySpan<float> yRawSpan = null;
            CpuTensor<float> y = null;
            Span<float> ySpan = null;
            int ySpanIndex = 0;
            if (File.Exists(yRawFileIfAny))
            {
                var yRawDataframe = Dataframe.Load(yRawFileIfAny, true, ',');
                yRaw = yRawDataframe.Tensor;
                yRawSpan = yRaw.AsReadonlyFloatCpuContent;
                y = new CpuTensor<float>(hyperParameters.Y_Shape(yRaw.Shape[0]));
                ySpan = y.AsFloatCpuSpan;
            }

            var divider = hyperParameters.RawCountToCount(1);

            for (int row = 0; row < x.Shape[0]; row++)
            {
                int rawRow = row / divider;
                int horizonId = hyperParameters.RowToHorizonId(row);
                int marketId = hyperParameters.RowToMarketId(row);

                //we load the row 'row' in 'x' tensor
                for (int col = 0; col < xRaw.Shape[1]; ++col)
                {
                    xSpan[xSpanIndex++] = xRawSpan[rawRow * xRaw.Shape[1] + col];
                }

                if (hyperParameters.MergeHorizonAndMarketIdInSameFeature)
                {
                    Debug.Assert(marketId >= 0);
                    Debug.Assert(horizonId >= 0);
                    xSpan[xSpanIndex++] = marketId*Natixis70Utils.HORIZON_NAMES.Length+ horizonId;
                }
                else
                {
                    if (marketId >= 0)
                    {
                        xSpan[xSpanIndex++] = marketId;
                    }
                    if (horizonId >= 0)
                    {
                        xSpan[xSpanIndex++] = horizonId;
                    }
                }
                if (yRaw != null)
                {
                    //we load the row 'row' in 'y' tensor
                    for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (HORIZON_NAMES.Length-1) : marketId); ++currentMarketId)
                    {
                        for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (HORIZON_NAMES.Length-1) : horizonId); ++currentHorizonId)
                        {
                            int rawColIndex = 1 + HORIZON_NAMES.Length * currentMarketId + currentHorizonId;
                            var yRawValue = yRawSpan[rawRow * yRaw.Shape[1] + rawColIndex];
                            if (hyperParameters.Normalization == Natixis70HyperParameters.normalize_enum.MINUS_MEAN_DIVIDE_BY_VOL)
                            {
                                var colStatistics = Y_RAW_statistics[rawColIndex-1];
                                yRawValue = (float) ((yRawValue - colStatistics.Average) / colStatistics.Volatility);
                            }
                            else if (hyperParameters.Normalization == Natixis70HyperParameters.normalize_enum.DIVIDE_BY_ABS_MEAN)
                            {
                                var absColStatistics = Y_RAW_abs_statistics[rawColIndex - 1];
                                yRawValue = (float)(yRawValue / absColStatistics.Average);
                            }
                            ySpan[ySpanIndex++] = yRawValue;
                        }
                    }
                }
            }
            Debug.Assert(xSpanIndex == x.Count);
            Debug.Assert(y == null || ySpanIndex == y.Count);
            return (x, y);
        }



        // ReSharper disable once UnusedParameter.Global
        public static Natixis70DataSet ValueOf(string xRawFile, string yRawFileIfAny, bool useBackgroundThreadToLoadNextMiniBatch, Action<string> log,
            Natixis70HyperParameters hyperParameters)
        {
            var (x, yIfAny) = LoadX_Y(xRawFile, yRawFileIfAny, hyperParameters);
            return new Natixis70DataSet(x, yIfAny, useBackgroundThreadToLoadNextMiniBatch, hyperParameters);
        }

        private Natixis70DataSet(CpuTensor<float> x, CpuTensor<float> yIfAny, bool useBackgroundThreadToLoadNextMiniBatch, Natixis70HyperParameters hyperParameters) 
            : base(
                x,
                yIfAny,
                hyperParameters.IsTryingToPredictErrors? "Natixis70Errors":"Natixis70", 
                Objective_enum.Regression,
                null,
                new[] {"NONE"},
                hyperParameters.ComputeFeatureNames(),
                useBackgroundThreadToLoadNextMiniBatch)
        {
            HyperParameters = hyperParameters;
        }

        

        /// <summary>
        /// the sub part of the original (and complete) Training Data Set used for Validation
        /// </summary>
        public Natixis70DataSet ValidationDataSet { get; set; }

        /// <summary>
        /// the original (and complete) Test Data Set for the Natixis70 challenge
        /// </summary>
        public Natixis70DataSet OriginalTestDataSet { get; set; }

      

        /// <summary>
        /// will save also the pid features, and the prediction file for the Train + Validation + Test datasets
        /// </summary>
        public override void SaveModelAndParameters(Network network, string modelFilePath, string parametersFilePath)
        {
            base.SaveModelAndParameters(network, modelFilePath, parametersFilePath);

            CreatePredictionFile(network, "train_predictions");
            ValidationDataSet?.CreatePredictionFile(network, "validation_predictions");
            OriginalTestDataSet?.CreatePredictionFile(network, "test_predictions");

            var embeddingLayer = network.Layers.FirstOrDefault(l => l is EmbeddingLayer);
            if (embeddingLayer == null)
            {
                return;
            }
            var cpuTensor = embeddingLayer.Weights.ToCpuFloat();
            cpuTensor.Save(Path.Combine(network.Config.LogDirectory, "pid_features_" + network.UniqueId + ".csv"),
                row => true, //TODO
                true,
                "pid;"+string.Join(";", Enumerable.Range(0,cpuTensor.Shape[0]).Select(i=>"feature_"+i))
                );
        }

        public void CreatePredictionFile(Network network, string subDirectory)
        {
            string filePath = Path.Combine(network.Config.LogDirectory, subDirectory, network.UniqueId + ".csv");
            var predictions = network.Predict(this, HyperParameters.BatchSize);
            HyperParameters.SavePredictions(predictions, filePath);
        }




        /// <summary>
        /// we'll save the network if we have reached a very small loss
        /// </summary>
        public override bool ShouldCreateSnapshotForEpoch(int epoch, Network network)
        {
            return    epoch >= 2
                   && network.CurrentEpochIsAbsolutelyBestInValidationLoss()
                   && !double.IsNaN(network.EpochData.Last().ValidationLoss)
                   && network.Config.AlwaysUseFullTestDataSetForLossAndAccuracy
                   && network.EpochData.Last().ValidationLoss < HyperParameters.MaxLossToSaveTheNetwork;
        }

        public override int Count => _x.Shape[0];

        public override int ElementIdToCategoryIndex(int elementId)
        {
            return -1;
        }

        public override double PercentageToUseForLossAndAccuracyFastEstimate => 0.0; //we do not compute any estimate

        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }

        public override string ToString()
        {
            var xShape = HyperParameters.X_Shape(Count);
            return Tensor.ShapeToString(xShape) + " => " + Tensor.ShapeToString(Y.Shape);
        }
    }
}
