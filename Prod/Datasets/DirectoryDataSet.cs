using SharpNet.CPU;
using SharpNet.Pictures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
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
        #endregion
        public override CpuTensor<float> Y { get; }


        public static DirectoryDataSet FromDirectory(string path, int nbCategories, CategoryHierarchy hierarchyIfAny)
        {
            var allFiles = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories).Where(PictureTools.IsPicture).ToList();
            var elementIdToDescription = allFiles.ToList();
            var elementIdToPaths = new List<List<string>>();
            var elementIdToCategoryIndex = new List<int>();
            foreach (var f in allFiles)
            {
                elementIdToPaths.Add(new List<string> {f});
                elementIdToCategoryIndex.Add(-1);
            }
            var categoryDescriptions = Enumerable.Range(0, nbCategories).Select(i=>i.ToString()).ToArray();
                
            return new DirectoryDataSet(
                elementIdToPaths,
                elementIdToDescription,
                elementIdToCategoryIndex,
                null,
                path, 
                3, 
                categoryDescriptions,
                DataSetBuilder.CancelMeanAndVolatilityForEachChannel,
                ResizeStrategyEnum.ResizeToTargetSize,
                hierarchyIfAny);

        }

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

            if (ResizeStrategy != ResizeStrategyEnum.ResizeToTargetSize)
            {
                throw new NotImplementedException("ResizeStrategy "+ ResizeStrategy+" is not supported");
            }

            if (meanAndVolatilityForEachChannel == null)
            {
                ComputeMeanAndVolatilityForEachChannel();
                throw new ArgumentException("please update mean and volatility for dataSet " + name);
            }
            //We compute Y if necessary
            Y = expectedYIfAny??CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryIndex, elementIdToDescription.Count, CategoryCount);
        }

        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            Debug.Assert(Channels == xBuffer.Shape[1]);
            int targetHeight = xBuffer.Shape[2];
            int targetWidth = xBuffer.Shape[3];
            
            var data = OriginalElementContent(elementId, targetHeight, targetWidth);

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

        //// ReSharper disable once UnusedMember.Global
        //public DirectoryDataSet CropBorder(bool skipIfFileAlreadyExists = true)
        //{
        //    var targetDirectory = _dataDirectory + "_cropped";
        //    if (!Directory.Exists(targetDirectory))
        //    {
        //        Directory.CreateDirectory(targetDirectory);
        //    }
        //    Logger.Info("Cropping " + Count + " elements and copying them in " + targetDirectory);
        //    int nbPerformed = 0;
        //    Parallel.For(0, Count, elementId => CropBorder(elementId, targetDirectory, skipIfFileAlreadyExists, ref nbPerformed));
        //    return new DirectoryDataSet(_descriptionPath, targetDirectory, Logger, Name, Channels, Height, Width, _categoryIndexToDescription, TODO, _ignoreZeroPixel, _computeCategoryIndexDescriptionFullName);
        //}
        //// ReSharper disable once UnusedMember.Global
        //public DirectoryDataSet Filter(Func<BitmapContent, bool> isIncluded, bool skipIfFileAlreadyExists = true)
        //{
        //    var targetDirectory = _dataDirectory + "_filter";
        //    if (!Directory.Exists(targetDirectory))
        //    {
        //        Directory.CreateDirectory(targetDirectory);
        //    }
        //    Logger.Info("Filtering " + Count + " elements and copying them in " + targetDirectory);
        //    int nbPerformed = 0;
        //    Parallel.For(0, Count, elementId => Filter(elementId, targetDirectory, isIncluded, skipIfFileAlreadyExists, ref nbPerformed));
        //    return new DirectoryDataSet(_descriptionPath, targetDirectory, Logger, Name, Channels, Height, Width, _categoryIndexToDescription, TODO, _ignoreZeroPixel, _computeCategoryIndexDescriptionFullName);
        //}
        //// ReSharper disable once UnusedMember.Global
        //public DirectoryDataSet Resize(int newWidth, int newHeight, bool skipIfFileAlreadyExists = true)
        //{
        //    var targetDirectory = _dataDirectory + "_resize_" + newWidth + "_" + newHeight;
        //    if (!Directory.Exists(targetDirectory))
        //    {
        //        Directory.CreateDirectory(targetDirectory);
        //    }
        //    Logger.Info("Resizing " + Count + " elements and copying them in " + targetDirectory);
        //    var nbPerformed = 0;
        //    Parallel.For(0, Count, elementId => Resize(elementId, targetDirectory, newWidth, newHeight, skipIfFileAlreadyExists, ref nbPerformed));
        //    return new DirectoryDataSet(_descriptionPath, targetDirectory, Logger, Name, Channels, Height, Width, _categoryIndexToDescription, TODO, _ignoreZeroPixel, _computeCategoryIndexDescriptionFullName);
        //}
        // ReSharper disable once UnusedMember.Global
        //public DirectoryDataSet MakeSquarePictures(bool alwaysUseBiggestSideForWidthSide, bool alwaysCropInsidePicture, Tuple<byte, byte, byte> fillingColor = null, bool skipIfFileAlreadyExists = true)
        //{
        //    fillingColor = fillingColor ?? Tuple.Create((byte)0, (byte)0, (byte)0);
        //    var targetDirectory = _dataDirectory + "_square";
        //    if (!Directory.Exists(targetDirectory))
        //    {
        //        Directory.CreateDirectory(targetDirectory);
        //    }
        //    Logger.Info("Making " + Count + " elements square pictures and copying them in " + targetDirectory);
        //    int nbPerformed = 0;
        //    Parallel.For(0, Count, elementId => MakeSquarePictures(elementId, targetDirectory, alwaysUseBiggestSideForWidthSide, alwaysCropInsidePicture, fillingColor, skipIfFileAlreadyExists, ref nbPerformed));
        //    return new DirectoryDataSet(_descriptionPath, targetDirectory, Logger, Name, Channels, Height, Width, _categoryIndexToDescription, TODO, _ignoreZeroPixel, _computeCategoryIndexDescriptionFullName);
        //}
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

        //private void MakeSquarePictures(int elementId, string targetDirectory, bool alwaysUseBiggestSideForWidthSide, bool alwaysCropInsidePicture, Tuple<byte, byte, byte> fillingColor, bool skipIfFileAlreadyExists, ref int nbPerformed)
        //{
        //    UpdateStatus(ref nbPerformed);
        //    var targetFileNames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
        //    if (skipIfFileAlreadyExists && AllFileExist(targetFileNames))
        //    {
        //        return;
        //    }
        //    var bitmapContent = OriginalElementContent(elementId);
        //    var squarePicture = bitmapContent.MakeSquarePictures(alwaysUseBiggestSideForWidthSide, alwaysCropInsidePicture, fillingColor);
        //    squarePicture.Save(targetFileNames);
        //}
        private static bool AllFileExist(IEnumerable<string> fileNames)
        {
            foreach (var filename in fileNames)
            {
                if (!File.Exists(filename))
                {
                    return false;
                }
            }
            return true;
        }
        //private void CropBorder(int elementId, string targetDirectory, bool skipIfFileAlreadyExists, ref int nbPerformed)
        //{
        //    UpdateStatus(ref nbPerformed);
        //    var targetFileNames = _elementIdToPaths[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
        //    if (skipIfFileAlreadyExists && AllFileExist(targetFileNames))
        //    {
        //        return;
        //    }
        //    var bitmapContent = OriginalElementContent(elementId);
        //    var cropped = bitmapContent.CropBorder();
        //    cropped.Save(targetFileNames);
        //}
        //private void Resize(int elementId, string targetDirectoryWithElements, int newWidth, int newHeight, bool skipIfFileAlreadyExists, ref int nbPerformed)
        //{
        //    UpdateStatus(ref nbPerformed);
        //    foreach(var subPath in _elementIdToPaths[elementId])
        //    {
        //        var srcFilename = Path.Combine(_dataDirectory, subPath);
        //        var targetFilename = Path.Combine(targetDirectoryWithElements, subPath);
        //        if (skipIfFileAlreadyExists && File.Exists(targetFilename))
        //        {
        //            continue;
        //        }
        //        using (var bmp = new Bitmap(srcFilename))
        //        {
        //            var resizedBmp = PictureTools.ResizeImage(bmp, newWidth, newHeight);
        //            PictureTools.SavePng(resizedBmp, targetFilename);
        //        }
        //    }
        //}
        //private void Filter(int elementId, string targetDirectory, Func<BitmapContent, bool> isIncluded, bool skipIfFileAlreadyExists, ref int nbPerformed)
        //{
        //    UpdateStatus(ref nbPerformed);
        //    var targetFileNames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
        //    if (skipIfFileAlreadyExists && AllFileExist(targetFileNames))
        //    {
        //        return;
        //    }
        //    var bitmapContent = OriginalElementContent(elementId);
        //    if (isIncluded(bitmapContent))
        //    {
        //        bitmapContent.Save(targetFileNames);
        //    }
        //}
        //private string StatsFileName => Path.Combine(_dataDirectory, "SharpNet.ini");
        //private void CreateStatsFile()
        //{
        //    Logger.Info("Creating stats file");
        //    var sb = new StringBuilder();
        //    //We compute Mean and volatility stats
        //    _meanAndVolatilityForEachChannel = new List<Tuple<float, float>>();
        //    _meanAndVolatilityForEachChannel.Clear();
        //    sb.Append(MEAN_VOLATILITY + "=");
        //    foreach (var meanVolatility in ComputeMeanAndVolatilityForEachChannel())
        //    {
        //        _meanAndVolatilityForEachChannel.Add(meanVolatility);
        //        sb.Append(meanVolatility.Item1.ToString(CultureInfo.InvariantCulture) + ";" + meanVolatility.Item2.ToString(CultureInfo.InvariantCulture) + ";");
        //    }
        //    sb.Append(Environment.NewLine);
        //    File.WriteAllText(StatsFileName, sb.ToString());
        //}
        //const string MEAN_VOLATILITY = "MEAN_VOLATILITY";
        //private bool LoadStatsFile()
        //{
        //    Log.Info("Loading stats file");
        //    if (!File.Exists(StatsFileName))
        //    {
        //        return false;
        //    }
        //    bool fileLoadedSuccessfully = false;
        //    var content = File.ReadAllLines(StatsFileName);
        //    _meanAndVolatilityForEachChannel = new List<Tuple<float, float>>();
        //    foreach (var line in content)
        //    {
        //        var splitted = line.Split('=');
        //        if (splitted.Length == 2 && splitted[0] == MEAN_VOLATILITY)
        //        {
        //            _meanAndVolatilityForEachChannel.Clear();
        //            var values = splitted[1].Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        //            for (int i = 0; i < values.Length; i += 2)
        //            {
        //                _meanAndVolatilityForEachChannel.Add(Tuple.Create(values[i], values[i + 1]));
        //            }
        //            fileLoadedSuccessfully = true;
        //        }
        //    }
        //    Log.Info(fileLoadedSuccessfully ? "stats file loaded successfully" : "fail to load stats file");
        //    return fileLoadedSuccessfully;
        //}

        public override BitmapContent OriginalElementContent(int elementId, int targetHeight, int targetWidth)
        {
            var elementPaths = _elementIdToPaths[elementId];
            if (elementPaths.Count == 1 && Channels == 3)
            {
                //single file containing all channels of the element
                using var bmp = new Bitmap(elementPaths[0]);
                if (targetHeight != -1 && targetWidth != -1 && (bmp.Width != targetWidth || bmp.Height != targetHeight))
                {
                    //we need to resize the bitmap 
                    Debug.Assert(ResizeStrategy == ResizeStrategyEnum.ResizeToTargetSize);
                    var interpolationMode = (bmp.Width * bmp.Height > targetWidth * targetHeight)
                        ? InterpolationMode.NearestNeighbor //we are reducing the image size
                        : InterpolationMode.Bicubic;        //we are increasing the image
                    using var resizedBitmap = PictureTools.ResizeImage(bmp, targetWidth, targetHeight, interpolationMode);
                    return BitmapContent.ValueFomSingleRgbBitmap(resizedBitmap);
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
            OriginalElementContent(elementId,-1,-1).UpdateWith_Sum_SumSquare_Count_For_Each_Channel(_sum_SumSquare_Count_For_Each_Channel);
        }
    }
}
