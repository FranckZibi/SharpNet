using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using SharpNet.CPU;
using SharpNet.LightGBM;
using SharpNet.Networks;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.Natixis70
{
    /// <summary>
    /// Network support for Natixis70 challenge
    /// </summary>
    public class Natixis70HyperParameters : NetworkBuilder
    {

        /// <summary>
        /// true if we want to predict all horizons (1d / 1w / 2w) at the same time
        /// false if we want to predict each horizon separately
        /// </summary>
        public bool TryToPredictAllHorizonAtTheSameTime = true;

        /// <summary>
        /// true if we want to predict all markets (VIX, EURUSD, etc.) at the same time
        /// false if we want to predict each market separately 
        /// </summary>
        public bool TryToPredictAllMarketsAtTheSameTime = false;

        /// <summary>
        /// normalize all label:
        ///     y = (y -average(y)) / volatility(y)
        /// </summary>
        public bool NormalizeAllLabels = false;


        public double PercentageInTraining = 0.8;


        public string[] ComputeFeatureNames()
        {
            var featureNames = new List<string>();
            for (int i = 0; i < Natixis70Utils.EMBEDDING_DIMENSION; ++i)
            {
                featureNames.Add("embed_" + i);
            }
            featureNames.AddRange(CategoricalFeatures());
            return featureNames.ToArray();
        }

        public List<string> CategoricalFeatures()
        {
            var categoricalFeatures = new List<string>();

            if (!TryToPredictAllMarketsAtTheSameTime)
            {
                categoricalFeatures.Add("marketId");
            }
            if (!TryToPredictAllHorizonAtTheSameTime)
            {
                categoricalFeatures.Add("horizonId");
            }
            return categoricalFeatures;
        }


        public void SavePredictions(CpuTensor<float> y, string path)
        {
            var sb = new StringBuilder();
            sb.Append(Natixis70Utils.PREDICTION_HEADER+Environment.NewLine);

            var ySpan = y.AsReadonlyFloatCpuContent;
            var ySpanIndex = 0;

            var yRaw = new CpuTensor<float>(YRaw_Shape(y.Shape[0]));
            var yRawSpan = yRaw.AsFloatCpuSpan;
            var divider = RawCountToCount(1);

            for (int row = 0; row < y.Shape[0]; ++row)
            {
                var rawRow = row / divider;
                yRawSpan[rawRow * yRaw.Shape[1]] = rawRow;
                int horizonId = RowToHorizonId(row);
                int marketId = RowToMarketId(row);
                //we load the row 'row' in 'yRaw' tensor
                for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (Natixis70Utils.HORIZON_NAMES.Length-1) : marketId); ++currentMarketId)
                {
                    for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (Natixis70Utils.HORIZON_NAMES.Length-1) : horizonId); ++currentHorizonId)
                    {
                        int rawColIndex = 1 + Natixis70Utils.HORIZON_NAMES.Length * currentMarketId + currentHorizonId;
                        var yValue = ySpan[ySpanIndex++];
                        if (Math.Abs(yValue)<1e-4)
                        {
                            yValue = 0;
                        }
                        yRawSpan[rawRow * yRaw.Shape[1] + rawColIndex] = yValue;
                    }
                }
            }
            new Dataframe(yRaw, Natixis70Utils.PREDICTION_HEADER.Split(','), "").Save(path);
        }



        public int RawCountToCount(int rawCount)
        {
            int count = rawCount;
            if (!TryToPredictAllHorizonAtTheSameTime)
            {
                count *= Natixis70Utils.HORIZON_NAMES.Length;
            }
            if (!TryToPredictAllMarketsAtTheSameTime)
            {
                count *= Natixis70Utils.MARKET_NAMES.Length;
            }
            return count;
        }
        public int CountToRawCount(int count)
        {
            int rawCount = count;
            if (!TryToPredictAllHorizonAtTheSameTime)
            {
                Debug.Assert(count % Natixis70Utils.HORIZON_NAMES.Length == 0);
                rawCount /= Natixis70Utils.HORIZON_NAMES.Length;
            }
            if (!TryToPredictAllMarketsAtTheSameTime)
            {
                Debug.Assert(count % Natixis70Utils.MARKET_NAMES.Length == 0);
                rawCount /= Natixis70Utils.MARKET_NAMES.Length;
            }
            return rawCount;
        }

        public int[] X_Shape(int xRowCount)
        {
            int xColCount = Natixis70Utils.EMBEDDING_DIMENSION;
            if (!TryToPredictAllHorizonAtTheSameTime)
            {
                xColCount += 1; //we'll have on more feature : the horizon to predict (1d / 1w / 2w)
            }
            if (!TryToPredictAllMarketsAtTheSameTime)
            {
                xColCount += 1; //we'll have on more feature : the market to predict (VIX, EURUSD, etc...)
            }
            return new[] { RawCountToCount(xRowCount), xColCount };
        }

        public int[] Y_Shape(int yRawCount)
        {
            int yColCount = 1;
            if (TryToPredictAllHorizonAtTheSameTime)
            {
                yColCount *= Natixis70Utils.HORIZON_NAMES.Length;
            }
            if (TryToPredictAllMarketsAtTheSameTime)
            {
                yColCount *= Natixis70Utils.MARKET_NAMES.Length;
            }
            return new[] { RawCountToCount(yRawCount), yColCount };
        }

        public int[] YRaw_Shape(int yCount)
        {
            return new[] { CountToRawCount(yCount), 1 + Natixis70Utils.MARKET_NAMES.Length* Natixis70Utils.HORIZON_NAMES.Length };
        }


        /// <summary>
        /// return the horizonId associated with row 'row', or -1 if the row is associated with all horizon ids
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        public int RowToHorizonId(int row)
        {
            if (TryToPredictAllHorizonAtTheSameTime)
            {
                return -1;
            }
            return row%Natixis70Utils.HORIZON_NAMES.Length;
        }

        /// <summary>
        /// return the marketId associated with row 'row', or -1 if the row is associated with all market ids
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        public int RowToMarketId(int row)
        {
            if (TryToPredictAllMarketsAtTheSameTime)
            {
                return -1;
            }
            if (!TryToPredictAllHorizonAtTheSameTime)
            {
                row /= Natixis70Utils.HORIZON_NAMES.Length;
            }

            return row % Natixis70Utils.MARKET_NAMES.Length;
        }
        // max value of the loss to consider saving the network
        public double MaxLossToSaveTheNetwork => double.MaxValue; //TODO

        public string[] predictionFilesIfComputeErrors = null;

        public bool IsTryingToPredictErrors => predictionFilesIfComputeErrors != null;

        // ReSharper disable once UnusedMember.Global
        public string DatasetName => IsTryingToPredictErrors ? "Natixis70Errors" : "Natixis70";
    }
}
