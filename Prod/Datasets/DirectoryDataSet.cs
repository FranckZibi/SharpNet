using SharpNet.CPU;
using SharpNet.Pictures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
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
        /// path to a file (or directory) with the description of the dataset elements (categories, etc.)
        /// </summary>
        private readonly string _descriptionPath;
        /// <summary>
        /// directory with all elements of the dataset. 
        /// </summary>
        private readonly string _dataDirectory;
        private readonly string[] _categoryIndexToDescription;
        private readonly bool _ignoreZeroPixel;
        private readonly Action<string, string, Logger, List<int>, List<string>, List<List<string>>> _computeCategoryIndexDescriptionFullName;
        /// <summary>
        /// For each element id, the list of path to files (starting from 'DataDirectory') used to construct it
        /// If this list contains 1 element:
        ///    this element contains all channels used to construct the element id
        ///  If this list contains several element:
        ///     each element is a single channel of the target element id
        ///     we'll need to stack all those channels to build the element id
        /// </summary>
        private readonly List<List<string>> _elementIdToSubPath = new List<List<string>>();
        private readonly List<string> _elementIdToDescription = new List<string>();
        private readonly List<int> _elementIdToCategoryIndex = new List<int>();
        #endregion
        /// <summary>
        /// height of each element in the directory
        /// -1 if element may have different height
        /// </summary>
        public override int Height { get; }
        /// <summary>
        /// width of each element in the directory
        /// -1 if element may have different width
        /// </summary>
        public override int Width { get; }

        public override CpuTensor<float> Y { get; }

        public DirectoryDataSet(string descriptionPath, string dataDirectory, Logger logger,
            string name, int channels, int height, int width, string[] categoryIndexToDescription,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            bool ignoreZeroPixel,
            Action<string, string, Logger, List<int>, List<string>, List<List<string>>>
                computeCategoryIndexDescriptionFullName)
            : base(name, channels, categoryIndexToDescription.Length, meanAndVolatilityForEachChannel, logger)
        {
            _descriptionPath = descriptionPath;
            Height = height;
            Width = width;
            _dataDirectory = dataDirectory ?? Path.GetDirectoryName(descriptionPath) ?? "";
            _categoryIndexToDescription = categoryIndexToDescription;
            _ignoreZeroPixel = ignoreZeroPixel;
            Debug.Assert(computeCategoryIndexDescriptionFullName != null);
            _computeCategoryIndexDescriptionFullName = computeCategoryIndexDescriptionFullName;
            _computeCategoryIndexDescriptionFullName(_descriptionPath, _dataDirectory, Logger, _elementIdToCategoryIndex, _elementIdToDescription,_elementIdToSubPath);
            //if (!LoadStatsFile())
            //{
            //    CreateStatsFile();
            //}
            if (meanAndVolatilityForEachChannel == null)
            {
                ComputeMeanAndVolatilityForEachChannel();
                throw new ArgumentException("please update mean and volatility for dataSet " + name);
            }
            //We compute Y 
            Y = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryIndex, _elementIdToCategoryIndex.Count, CategoryCount);
        }
        // ReSharper disable once UnusedMember.Global
        public static void DefaultCompute_CategoryIndex_Description_FullName(
            string csvFileName,
            string directoryWithElements,
            Logger logger,
            List<int> elementIdToCategoryIndex,
            List<string> elementIdToDescription,
            List<List<string>> elementIdToSubPath
            )
        {
            elementIdToCategoryIndex.Clear();
            elementIdToDescription.Clear();
            elementIdToSubPath.Clear();

            if (!File.Exists(csvFileName))
            {
                logger.Info("missing file " + csvFileName);
                throw new ArgumentException("missing file " + csvFileName);
            }

            //we retrieve all files in the directory containing the elements
            var fileNameWithoutExtensionToSubPath = new Dictionary<string, string>();
            foreach (var fileInfo in new DirectoryInfo(directoryWithElements).GetFiles())
            {
                fileNameWithoutExtensionToSubPath[Path.GetFileNameWithoutExtension(fileInfo.Name)] = fileInfo.Name;
            }

            var lines = File.ReadAllLines(csvFileName);
            for (var index = 0; index < lines.Length; index++)
            {
                var splittedLine = lines[index].Split(',', ';').ToArray();
                var description = splittedLine[0];

                int categoryIndex = -1; //unknown categoryIndex
                if (splittedLine.Length == 2)
                { 
                    var categoryIndexAsString = splittedLine[1];
                    if (!int.TryParse(categoryIndexAsString, out categoryIndex) || categoryIndex < 0)
                    {
                        if (index == 0)
                        {
                            logger.Debug("ignoring (header) first line: " + lines[index]);
                        }
                        else
                        {
                            logger.Info("invalid categoryIndex in line: " + lines[index]);
                            throw new ArgumentException("invalid categoryIndex in line: " + lines[index]);
                        }
                        continue;
                    }
                }
                var elementFileNameWithoutExtension = Path.GetFileNameWithoutExtension(description) ?? "";
                if (!fileNameWithoutExtensionToSubPath.ContainsKey(elementFileNameWithoutExtension))
                {
                    logger.Debug("WARNING: no matching file for line: " + lines[index]);
                    continue;
                }
                elementIdToCategoryIndex.Add(categoryIndex);
                elementIdToDescription.Add(description);
                elementIdToSubPath.Add(new List<string> { fileNameWithoutExtensionToSubPath[elementFileNameWithoutExtension] });
            }
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            var data = OriginalElementContent(elementId);
            var xBufferContent = xBuffer.SpanContent;
            for (int channel = 0; channel < data.GetChannels(); ++channel)
            {
                for (int row = 0; row < data.GetHeight(); ++row)
                {
                    for (int col = 0; col < data.GetWidth(); ++col)
                    {
                        var val = (double)data.Get(channel, row, col);
                        // ReSharper disable once RedundantLogicalConditionalExpressionOperand
                        if (_ignoreZeroPixel && data.IsZeroPixel(row, col))
                        {
                            // no normalization
                        }
                        else
                        {
                            val = (val - OriginalChannelMean(channel)) / OriginalChannelVolatility(channel);
                        }
                        var bufferIdx = xBuffer.Idx(indexInBuffer, channel, row, col);
                        xBufferContent[bufferIdx] = (float)val;
                    }
                }
            }
            var categoryIndex = ElementIdToCategoryIndex(elementId);
            for (int cat = 0; cat < CategoryCount; ++cat)
            {
                yBuffer?.Set(indexInBuffer, cat, (cat==categoryIndex)?1f: 0f);
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

        private List<Tuple<float, float>> ComputeMeanAndVolatilityForEachChannel()
        {
            const int DistinctValuesToComputeInEachChannel = 3; //sum + sumSquare + count
            var sumSumSquareCountForEachChannel = new float[Channels * DistinctValuesToComputeInEachChannel];
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => UpdateWith_Sum_SumSquare_Count_For_Each_Channel(elementId, sumSumSquareCountForEachChannel, _ignoreZeroPixel, ref nbPerformed));
            return Sum_SumSquare_Count_to_ComputeMeanAndVolatilityForEachChannel(sumSumSquareCountForEachChannel);
        }

        private void MakeSquarePictures(int elementId, string targetDirectory, bool alwaysUseBiggestSideForWidthSide, bool alwaysCropInsidePicture, Tuple<byte, byte, byte> fillingColor, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            var targetFileNames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
            if (skipIfFileAlreadyExists && AllFileExist(targetFileNames))
            {
                return;
            }
            var bitmapContent = OriginalElementContent(elementId);
            var squarePicture = bitmapContent.MakeSquarePictures(alwaysUseBiggestSideForWidthSide, alwaysCropInsidePicture, fillingColor);
            squarePicture.Save(targetFileNames);
        }
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
        private void CropBorder(int elementId, string targetDirectory, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            var targetFileNames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
            if (skipIfFileAlreadyExists && AllFileExist(targetFileNames))
            {
                return;
            }
            var bitmapContent = OriginalElementContent(elementId);
            var cropped = bitmapContent.CropBorder();
            cropped.Save(targetFileNames);
        }
        private void Resize(int elementId, string targetDirectoryWithElements, int newWidth, int newHeight, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            foreach(var subPath in _elementIdToSubPath[elementId])
            {
                var srcFilename = Path.Combine(_dataDirectory, subPath);
                var targetFilename = Path.Combine(targetDirectoryWithElements, subPath);
                if (skipIfFileAlreadyExists && File.Exists(targetFilename))
                {
                    continue;
                }
                using (var bmp = new Bitmap(srcFilename))
                {
                    var resizedBmp = PictureTools.ResizeImage(bmp, newWidth, newHeight);
                    PictureTools.SavePng(resizedBmp, targetFilename);
                }
            }
        }
        private void Filter(int elementId, string targetDirectory, Func<BitmapContent, bool> isIncluded, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            var targetFileNames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
            if (skipIfFileAlreadyExists && AllFileExist(targetFileNames))
            {
                return;
            }
            var bitmapContent = OriginalElementContent(elementId);
            if (isIncluded(bitmapContent))
            {
                bitmapContent.Save(targetFileNames);
            }
        }
        private string StatsFileName => Path.Combine(_dataDirectory, "SharpNet.ini");
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
        const string MEAN_VOLATILITY = "MEAN_VOLATILITY";
        private bool LoadStatsFile()
        {
            Logger.Info("Loading stats file");
            if (!File.Exists(StatsFileName))
            {
                return false;
            }
            bool fileLoadedSuccessfully = false;
            var content = File.ReadAllLines(StatsFileName);
            _meanAndVolatilityForEachChannel = new List<Tuple<float, float>>();
            foreach (var line in content)
            {
                var splitted = line.Split('=');
                if (splitted.Length == 2 && splitted[0] == MEAN_VOLATILITY)
                {
                    _meanAndVolatilityForEachChannel.Clear();
                    var values = splitted[1].Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();
                    for (int i = 0; i < values.Length; i += 2)
                    {
                        _meanAndVolatilityForEachChannel.Add(Tuple.Create(values[i], values[i + 1]));
                    }
                    fileLoadedSuccessfully = true;
                }
            }
            Logger.Info(fileLoadedSuccessfully ? "stats file loaded successfully" : "fail to load stats file");
            return fileLoadedSuccessfully;
        }

        public override BitmapContent OriginalElementContent(int elementId)
        {
            var fullNames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(_dataDirectory, subPath)).ToList();
            if (fullNames.Count == 1 && Channels == 3)
            {
                //single file containing all channels of the element
                return BitmapContent.ValueFomSingleRgbBitmap(fullNames[0]);
            }
            else
            {
                Debug.Assert(Channels == fullNames.Count);
                //each file contains 1 channel of the element
                return BitmapContent.ValueFromSeveralSingleChannelBitmaps(fullNames);
            }
        }
    }
}
