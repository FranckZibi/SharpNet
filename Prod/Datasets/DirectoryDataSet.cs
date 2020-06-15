﻿using SharpNet.CPU;
using SharpNet.Pictures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
// ReSharper disable UnusedMember.Local

namespace SharpNet.Datasets
{
    public class DirectoryDataSet : AbstractDataSet
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
        private readonly List<string> _elementIdToDescription = new List<string>();
        private readonly List<int> _elementIdToCategoryIndex;
        private readonly Random _rand = new Random(0);
        #endregion
        public override CpuTensor<float> Y { get; }


        #region constructor
        public DirectoryDataSet(
            List<List<string>> elementIdToPaths, 
            List<string> elementIdToDescription, 
            List<int> elementIdToCategoryIndex,
            CpuTensor<float> expectedYIfAny,
            string name, int channels, string[] categoryDescriptions,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            ResizeStrategyEnum resizeStrategy,
            CategoryHierarchy hierarchyIfAny)
            : base(name, channels, categoryDescriptions, meanAndVolatilityForEachChannel, resizeStrategy, hierarchyIfAny)
        {
            _elementIdToPaths.AddRange(elementIdToPaths);
            _elementIdToDescription.AddRange(elementIdToDescription);
            _elementIdToCategoryIndex = elementIdToCategoryIndex?.ToList();

            if (meanAndVolatilityForEachChannel == null)
            {
                ComputeMeanAndVolatilityForEachChannel();
                throw new ArgumentException("please update mean and volatility for dataSet " + name);
            }
            //We compute Y if necessary
            Y = expectedYIfAny??CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryIndex, elementIdToDescription.Count, CategoryCount);
        }
        #endregion

        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            Debug.Assert(Channels == xBuffer.Shape[1]);
            int targetHeight = xBuffer.Shape[2];
            int targetWidth = xBuffer.Shape[3];
            
            var data = OriginalElementContent(elementId, targetHeight, targetWidth, withDataAugmentation);

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
            var categoryIndex = ElementIdToCategoryIndex(elementId);
            if (categoryIndex == -1)
            {
                for (int cat = 0; cat < CategoryCount; ++cat)
                {
                    yBuffer?.Set(indexInBuffer, cat, Y.Get(elementId, cat));
                }
            }
            else
            {
                for (int cat = 0; cat < CategoryCount; ++cat)
                {
                    yBuffer?.Set(indexInBuffer, cat, (cat == categoryIndex) ? 1f : 0f);
                }
            }
        }

        public override int Count => _elementIdToCategoryIndex.Count;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _elementIdToCategoryIndex[elementId];
        }
        public override string ElementIdToDescription(int elementId)
        {
            return _elementIdToDescription[elementId];
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return _elementIdToPaths[elementId][0];
        }


        // ReSharper disable once UnusedMethodReturnValue.Local
        private List<Tuple<float, float>> ComputeMeanAndVolatilityForEachChannel()
        {
            const int DistinctValuesToComputeInEachChannel = 3; //sum + sumSquare + count
            var sumSumSquareCountForEachChannel = new float[Channels * DistinctValuesToComputeInEachChannel];
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => UpdateWith_Sum_SumSquare_Count_For_Each_Channel(elementId, sumSumSquareCountForEachChannel, ref nbPerformed));
            return Sum_SumSquare_Count_to_ComputeMeanAndVolatilityForEachChannel(sumSumSquareCountForEachChannel, Channels);
        }

        /// <summary>
        /// extract the original element 'elementId' from disk, resize it if needed, and returns it
        /// </summary>
        /// <param name="elementId"></param>
        /// <param name="targetHeight">the mandatory height of the output bitmap (or -1 if we should keep the same height of the original image)</param>
        /// <param name="targetWidth">the mandatory width of the output bitmap (or -1 if we should keep the same width of the original image)</param>
        /// <param name="withDataAugmentation"></param>
        /// <returns></returns>
        public override BitmapContent OriginalElementContent(int elementId, int targetHeight, int targetWidth, bool withDataAugmentation)
        {
            var elementPaths = _elementIdToPaths[elementId];
            if (elementPaths.Count == 1 && Channels == 3)
            {
                //single file containing all channels of the element
                using var bmp = new Bitmap(elementPaths[0]);
                if (targetHeight != -1 && targetWidth != -1 && (bmp.Width != targetWidth || bmp.Height != targetHeight))
                {
                    var interpolationMode = (bmp.Width * bmp.Height > targetWidth * targetHeight)
                        ? InterpolationMode.NearestNeighbor //we are reducing the image size
                        : InterpolationMode.Bicubic; //we are increasing the image

                    //we need to resize the bitmap 
                    if (ResizeStrategy == ResizeStrategyEnum.ResizeToTargetSize)
                    {
                        using var resizedBitmap = PictureTools.ResizeImage(bmp, targetWidth, targetHeight, interpolationMode);
                        return BitmapContent.ValueFomSingleRgbBitmap(resizedBitmap);
                    }
                    else if (ResizeStrategy == ResizeStrategyEnum.BiggestCropInOriginalImageToKeepSameProportion)
                    {
                        //const double toleranceInPercentage = 0.05;
                        const double toleranceInPercentage = 0.00;
                        var targetRatio = targetWidth / (double)targetHeight;
                        var originalRatio = bmp.Width / (double)bmp.Height;
                        Rectangle croppedRectangle;
                        if (targetRatio >= originalRatio)
                        {
                            int cropWidth = bmp.Width;
                            int cropHeight = Math.Min((int)((1 + toleranceInPercentage) * (cropWidth / targetRatio)), bmp.Height);
                            int firstTopRow = (withDataAugmentation)?_rand.Next(0, bmp.Height - cropHeight+1):(bmp.Height - cropHeight) / 2;
                            croppedRectangle =  new Rectangle(0, firstTopRow, cropWidth, cropHeight);
                        }
                        else
                        {
                            int cropHeight = bmp.Height;
                            int cropWidth = Math.Min((int)((1 + toleranceInPercentage) * (cropHeight * targetRatio)), bmp.Width);
                            int firstLeftColumn = (withDataAugmentation) ? _rand.Next(0, bmp.Width - cropWidth+ 1) : (bmp.Width - cropWidth) / 2;
                            croppedRectangle = new Rectangle(firstLeftColumn, 0, cropWidth, cropHeight);
                        }
                        using var croppedBitmap = PictureTools.CropImage(bmp, croppedRectangle);
                        using var resizedBitmap = PictureTools.ResizeImage(croppedBitmap, targetWidth, targetHeight, interpolationMode);
                        return BitmapContent.ValueFomSingleRgbBitmap(resizedBitmap);
                    }

                    throw new NotImplementedException("ResizeStrategy " + ResizeStrategy + " is not supported");
                }
                return BitmapContent.ValueFomSingleRgbBitmap(bmp);
            }

            Debug.Assert(Channels == elementPaths.Count);
            //each file contains 1 channel of the element
            return BitmapContent.ValueFromSeveralSingleChannelBitmaps(elementPaths);
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
                var volatility = (float)Math.Sqrt(Math.Max(0, variance));
                Log.Info("Mean and volatility for channel#" + channel + " : " + mean.ToString(CultureInfo.InvariantCulture) + " ; " + volatility.ToString(CultureInfo.InvariantCulture));
                result.Add(Tuple.Create(mean, volatility));
            }
            return result;
        }
        private void UpdateWith_Sum_SumSquare_Count_For_Each_Channel(int elementId, float[] _sum_SumSquare_Count_For_Each_Channel, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            OriginalElementContent(elementId,-1,-1, false).UpdateWith_Sum_SumSquare_Count_For_Each_Channel(_sum_SumSquare_Count_For_Each_Channel);
        }
    }
}
