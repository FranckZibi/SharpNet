using SharpNet.CPU;
using SharpNet.Pictures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;

// ReSharper disable UnusedMember.Local

namespace SharpNet.Datasets
{
    public class DirectoryDataSet : DataSet
    {
        #region private fields
        /// <summary>
        /// For each element id, the list of path to files used to construct it
        /// If this list contains 1 element:
        ///    this element contains all channels used to construct the element id
        ///  If this list contains several element:
        ///     each element is a single channel of the target element id
        ///     we'll need to stack all those channels to build the element id
        /// </summary>
        private readonly List<List<string>> _elementIdToPaths = new List<List<string>>();
        private readonly List<int> _elementIdToCategoryIndex;
        private readonly Random _rand = new Random(0);
        #endregion

        public override CpuTensor<float> Y => Y_DirectoryDataSet;
        public CpuTensor<float> Y_DirectoryDataSet { get; }

        public static DirectoryDataSet FromFiles(
            List<string> picturePaths,
            int categoryCount,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            ResizeStrategyEnum resizeStrategy)
        {
            var elementIdToPaths = new List<List<string>>();
            var rowInTargetFormatPredictionToID = new List<string>();
            var elementIdToCategoryIndex = new List<int>();
            foreach (var p in picturePaths)
            {
                elementIdToPaths.Add( new List<string> {p});
                rowInTargetFormatPredictionToID.Add(p);
                elementIdToCategoryIndex.Add(-1);
            }
            return new DirectoryDataSet(
                elementIdToPaths,
                elementIdToCategoryIndex,
                null,
                "FromFiles", 
                Objective_enum.Classification,
                3, 
                Enumerable.Range(0, categoryCount).Select(i=>i.ToString()).ToArray(),
                meanAndVolatilityForEachChannel,
                resizeStrategy,
                null,
                rowInTargetFormatPredictionToID.ToArray());
        }

        #region constructor
        public DirectoryDataSet(
            List<List<string>> elementIdToPaths,
            List<int> elementIdToCategoryIndex,
            CpuTensor<float> expectedYIfAny,
            string name,
            Objective_enum objective,
            int channels,
            string[] categoryDescriptions,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            ResizeStrategyEnum resizeStrategy,
            string[] featureNames,
            string[] elementIdToID)
            : base(name,
                objective, 
                meanAndVolatilityForEachChannel, 
                resizeStrategy,
                featureNames?? new string[0],
                new string[0],
                "",
                elementIdToID,
                ',')
        {
            _elementIdToPaths.AddRange(elementIdToPaths);
            _elementIdToCategoryIndex = elementIdToCategoryIndex?.ToList();

            if (meanAndVolatilityForEachChannel == null)
            {
                meanAndVolatilityForEachChannel = ComputeMeanAndVolatilityForEachChannel(channels);
                throw new ArgumentException("please update mean and volatility for dataSet " + name);
            }

            //We compute Y field
            if (expectedYIfAny != null)
            {
                Y_DirectoryDataSet = expectedYIfAny;
            }
            else
            {
                Debug.Assert(categoryDescriptions != null);
                Y_DirectoryDataSet = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryIndex, Count, categoryDescriptions.Length);
            }
        }
        #endregion

        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer,
            CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
        {
            int channels = xBuffer.Shape[1];
            int targetHeight = xBuffer.Shape[2];
            int targetWidth = xBuffer.Shape[3];

            var data = OriginalElementContent(elementId, channels, targetHeight, targetWidth, withDataAugmentation, isTraining);
            if (data == null)
            {
                return;
            }

            var xBufferContent = xBuffer.SpanContent;
            for (int channel = 0; channel < data.GetChannels(); ++channel)
            {
                for (int row = 0; row < data.GetHeight(); ++row)
                {
                    for (int col = 0; col < data.GetWidth(); ++col)
                    {
                        var val = (double)data.Get(channel, row, col);
                        val = (val - OriginalChannelMean(channel)) / OriginalChannelVolatility(channel);
                        var bufferIdx = xBuffer.Idx(indexInBuffer, channel, row, col);
                        xBufferContent[bufferIdx] = (float)val;
                    }
                }
            }

            if (yBuffer != null)
            {
                Y_DirectoryDataSet.CopyTo(Y_DirectoryDataSet.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
            }
        }
        public override bool CanBeSavedInCSV => false;
        public override int Count => _elementIdToPaths.Count;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _elementIdToCategoryIndex[elementId];
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return _elementIdToPaths[elementId][0];
        }


        // ReSharper disable once UnusedMethodReturnValue.Local
        private List<Tuple<float, float>> ComputeMeanAndVolatilityForEachChannel(int channels)
        {
            const int DistinctValuesToComputeInEachChannel = 3; //sum + sumSquare + count
            var sumSumSquareCountForEachChannel = new float[channels * DistinctValuesToComputeInEachChannel];
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => UpdateWith_Sum_SumSquare_Count_For_Each_Channel(elementId, channels, sumSumSquareCountForEachChannel, ref nbPerformed));
            return Sum_SumSquare_Count_to_ComputeMeanAndVolatilityForEachChannel(sumSumSquareCountForEachChannel, channels);
        }

        /// <summary>
        /// extract the original element 'elementId' from disk, resize it if needed, and returns it
        /// </summary>
        /// <param name="elementId"></param>
        /// <param name="channels"></param>
        /// <param name="targetHeight">the mandatory height of the output bitmap (or -1 if we should keep the same height of the original image)</param>
        /// <param name="targetWidth">the mandatory width of the output bitmap (or -1 if we should keep the same width of the original image)</param>
        /// <param name="withDataAugmentation"></param>
        /// <param name="isTraining"></param>
        /// <returns></returns>
        public override BitmapContent OriginalElementContent(int elementId, int channels, int targetHeight, int targetWidth, bool withDataAugmentation, bool isTraining)
        {
            try
            {
                var elementPaths = _elementIdToPaths[elementId];
                if (elementPaths.Count == 1 && channels == 3)
                {
                    //single file containing all channels of the element
                    var path = elementPaths[0];
                    if (!System.IO.File.Exists(path))
                    {
                        return null;
                    }

                    var bmpSize = PictureTools.ImageSize(path);
                    if (targetHeight != -1 && targetWidth != -1 && (bmpSize.Width != targetWidth || bmpSize.Height != targetHeight))
                    {
                        //we need to resize the bitmap 
                        if (ResizeStrategy == ResizeStrategyEnum.ResizeToTargetSize)
                        {
                            return BitmapContent.Resize(path, targetWidth, targetHeight);
                        }
                        else if (ResizeStrategy == ResizeStrategyEnum.BiggestCropInOriginalImageToKeepSameProportion)
                        {
                            //const double toleranceInPercentage = 0.05;
                            const double toleranceInPercentage = 0.00;
                            var targetRatio = targetWidth / (double)targetHeight;
                            var originalRatio = bmpSize.Width / (double)bmpSize.Height;
                            System.Drawing.Rectangle croppedRectangle;
                            if (targetRatio >= originalRatio)
                            {
                                int cropWidth = bmpSize.Width;
                                int cropHeight = Math.Min((int)((1 + toleranceInPercentage) * (cropWidth / targetRatio)), bmpSize.Height);
                                int firstTopRow = (withDataAugmentation)?_rand.Next(0, bmpSize.Height - cropHeight+1):(bmpSize.Height - cropHeight) / 2;
                                croppedRectangle =  new System.Drawing.Rectangle(0, firstTopRow, cropWidth, cropHeight);
                            }
                            else
                            {
                                int cropHeight = bmpSize.Height;
                                int cropWidth = Math.Min((int)((1 + toleranceInPercentage) * (cropHeight * targetRatio)), bmpSize.Width);
                                int firstLeftColumn = (withDataAugmentation) ? _rand.Next(0, bmpSize.Width - cropWidth+ 1) : (bmpSize.Width - cropWidth) / 2;
                                croppedRectangle = new System.Drawing.Rectangle(firstLeftColumn, 0, cropWidth, cropHeight);
                            }
                            return BitmapContent.CropAndResize(path, croppedRectangle, targetWidth, targetHeight);
                        }

                        throw new NotImplementedException("ResizeStrategy " + ResizeStrategy + " is not supported");
                    }
                    return BitmapContent.ValueFomSingleRgbBitmap(path);
                }
                //each file contains 1 channel of the element
                return BitmapContent.ValueFromSeveralSingleChannelBitmaps(elementPaths);
            }
            catch (Exception e)
            {
                Log.Error("Fail to load "+string.Join(" ", _elementIdToPaths[elementId]), e);
                return null;
            }
        }
        private static List<Tuple<float, float>> Sum_SumSquare_Count_to_ComputeMeanAndVolatilityForEachChannel(float[] sumSumSquareCountForEachChannel, int channelCount)
        {
            const int DistinctValuesToComputeInEachChannel = 3; //sum + sumSquare + count
            var result = new List<Tuple<float, float>>();
            for (int channel = 0; channel < channelCount; ++channel)
            {
                var sum = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel];
                var sumSquare = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel + 1];
                var count = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel + 2];
                var mean = (sum / count);
                var variance = (sumSquare / count) - mean * mean;
                var volatility = MathF.Sqrt(Math.Max(0, variance));
                Log.Info("Mean and volatility for channel#" + channel + " : " + mean.ToString(CultureInfo.InvariantCulture) + " ; " + volatility.ToString(CultureInfo.InvariantCulture));
                result.Add(Tuple.Create(mean, volatility));
            }
            return result;
        }
        private void UpdateWith_Sum_SumSquare_Count_For_Each_Channel(int elementId, int channels, float[] _sum_SumSquare_Count_For_Each_Channel, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            OriginalElementContent(elementId, channels ,- 1,-1, false, false).UpdateWith_Sum_SumSquare_Count_For_Each_Channel(_sum_SumSquare_Count_For_Each_Channel);
        }
    }
}
