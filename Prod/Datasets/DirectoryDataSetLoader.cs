using SharpNet.CPU;
using SharpNet.Pictures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SharpNet.Datasets
{
    public class DirectoryDataSetLoader : AbstractDataSetLoader
    {
        #region private fields
        private readonly string _csvFileName;
        private readonly string _dataDirectory;
        private readonly Logger _logger;
        private readonly string[] _categoryIdToDescription;
        private readonly bool _ignoreZeroPixel;
        private readonly Action<string, string, List<int>, List<string>, List<List<string>>, Logger> _computeCategoryIdDescriptionFullName;
        /// <summary>
        /// For each element id, the list of path to files (starting from 'DataDirectory') used to construct it
        /// If this list contains 1 element:
        ///    this element contains all channels used to construct the element id
        ///  If this list contains several element:
        ///     each element is a single channel of the target element id
        ///     we'll need ot stack all those channels to build the element id
        /// </summary>
        private readonly List<List<string>> _elementIdToSubPath = new List<List<string>>();
        private readonly List<string> _elementIdToDescription = new List<string>();
        private readonly List<int> _elementIdToCategoryId = new List<int>();
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel = new List<Tuple<float, float>>();
        #endregion
        public override int Height { get; }
        public override int Width { get; }

        public override CpuTensor<float> Y { get; }

        public DirectoryDataSetLoader(string csvFileName, string dataDirectory, Logger logger,
            int channels, int height, int width, string[] categoryIdToDescription,
            bool ignoreZeroPixel,
            Action<string,string,List<int>,List<string>,List<List<string>>,Logger> Compute_CategoryId_Description_FullName
            )
            : base(channels, categoryIdToDescription.Length)
        {
            _csvFileName = csvFileName;
            _logger = logger ?? Logger.ConsoleLogger;
            Height = height;
            Width = width;
            _dataDirectory = dataDirectory ?? Path.GetDirectoryName(csvFileName) ?? "";
            _categoryIdToDescription = categoryIdToDescription;
            _ignoreZeroPixel = ignoreZeroPixel;
            _computeCategoryIdDescriptionFullName = Compute_CategoryId_Description_FullName;
            Debug.Assert(Compute_CategoryId_Description_FullName != null);
            _computeCategoryIdDescriptionFullName(
                _csvFileName, _dataDirectory, _elementIdToCategoryId, _elementIdToDescription,
                _elementIdToSubPath, _logger);
            if (!LoadStatsFile())
            {
                CreateStatsFile();
            }
            //We compute Y 
            Y = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryId, _elementIdToCategoryId.Count, Categories);
        }
        public static void DefaultCompute_CategoryId_Description_FullName(
            string csvFileName,
            string directoryWithElements,
            List<int> elementIdToCategoryId,
            List<string> elementIdToDescription,
            List<List<string>> elementIdToSubPath, 
            Logger _logger)
        {
            elementIdToCategoryId.Clear();
            elementIdToDescription.Clear();
            elementIdToSubPath.Clear();

            if (!File.Exists(csvFileName))
            {
                _logger.Info("missing file " + csvFileName);
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
                var lineContent = lines[index].Split(',', ';').ToArray();
                var description = lineContent[0];

                int categoryId = -1; //unknown categoryId
                if (lineContent.Length == 2)
                { 
                    var categoryIdAsString = lineContent[1];
                    if (!int.TryParse(categoryIdAsString, out categoryId) || categoryId < 0)
                    {
                        if (index == 0)
                        {
                            _logger.Debug("ignoring (header) first line: " + lines[index]);
                        }
                        else
                        {
                            _logger.Info("invalid categoryId in line: " + lines[index]);
                            throw new ArgumentException("invalid categoryId in line: " + lines[index]);
                        }
                        continue;
                    }
                }
                var elementFileNameWithoutExtension = Path.GetFileNameWithoutExtension(description) ?? "";
                if (!fileNameWithoutExtensionToSubPath.ContainsKey(elementFileNameWithoutExtension))
                {
                    _logger.Debug("WARNING: no matching file for line: " + lines[index]);
                    continue;
                }
                elementIdToCategoryId.Add(categoryId);
                elementIdToDescription.Add(description);
                elementIdToSubPath.Add(new List<string> { fileNameWithoutExtensionToSubPath[elementFileNameWithoutExtension] });
            }
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> buffer)
        {
            var data = NewBitmapContent(elementId);
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
                            val = (val - ChannelMean(channel)) / ChannelVolatility(channel);
                        }
                        var bufferIdx = buffer.Idx(indexInBuffer, channel, row, col);
                        buffer.Content[bufferIdx] = (float)val;
                    }
                }
            }
        }
        // ReSharper disable once UnusedMember.Global
        public DirectoryDataSetLoader CropBorder(bool skipIfFileAlreadyExists = true)
        {
            var targetDirectory = _dataDirectory + "_cropped";
            if (!Directory.Exists(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            _logger.Info("Cropping " + Count + " elements and copying them in " + targetDirectory);
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => CropBorder(elementId, targetDirectory, skipIfFileAlreadyExists, ref nbPerformed));
            return new DirectoryDataSetLoader(_csvFileName, targetDirectory, _logger, Channels, Height, Width, _categoryIdToDescription, _ignoreZeroPixel, _computeCategoryIdDescriptionFullName);
        }
        // ReSharper disable once UnusedMember.Global
        public DirectoryDataSetLoader Filter(Func<BitmapContent, bool> isIncluded, bool skipIfFileAlreadyExists = true)
        {
            var targetDirectory = _dataDirectory + "_filter";
            if (!Directory.Exists(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            _logger.Info("Filtering " + Count + " elements and copying them in " + targetDirectory);
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => Filter(elementId, targetDirectory, isIncluded, skipIfFileAlreadyExists, ref nbPerformed));
            return new DirectoryDataSetLoader(_csvFileName, targetDirectory, _logger, Channels, Height, Width, _categoryIdToDescription, _ignoreZeroPixel, _computeCategoryIdDescriptionFullName);
        }
        // ReSharper disable once UnusedMember.Global
        public DirectoryDataSetLoader Resize(int newWidth, int newHeight, bool skipIfFileAlreadyExists = true)
        {
            var targetDirectory = _dataDirectory + "_resize_" + newWidth + "_" + newHeight;
            if (!Directory.Exists(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            _logger.Info("Resizing " + Count + " elements and copying them in " + targetDirectory);
            var nbPerformed = 0;
            Parallel.For(0, Count, elementId => Resize(elementId, targetDirectory, newWidth, newHeight, skipIfFileAlreadyExists, ref nbPerformed));
            return new DirectoryDataSetLoader(_csvFileName, targetDirectory, _logger, Channels, Height, Width, _categoryIdToDescription, _ignoreZeroPixel, _computeCategoryIdDescriptionFullName);
        }
        // ReSharper disable once UnusedMember.Global
        public DirectoryDataSetLoader MakeSquarePictures(bool alwaysUseBiggestSideForWidthSide, bool alwaysCropInsidePicture, Tuple<byte, byte, byte> fillingColor = null, bool skipIfFileAlreadyExists = true)
        {
            fillingColor = fillingColor ?? Tuple.Create((byte)0, (byte)0, (byte)0);
            var targetDirectory = _dataDirectory + "_square";
            if (!Directory.Exists(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            _logger.Info("Making " + Count + " elements square pictures and copying them in " + targetDirectory);
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => MakeSquarePictures(elementId, targetDirectory, alwaysUseBiggestSideForWidthSide, alwaysCropInsidePicture, fillingColor, skipIfFileAlreadyExists, ref nbPerformed));
            return new DirectoryDataSetLoader(_csvFileName, targetDirectory, _logger, Channels, Height, Width, _categoryIdToDescription, _ignoreZeroPixel, _computeCategoryIdDescriptionFullName);
        }
        public override int Count => _elementIdToCategoryId.Count;
        public override int ElementIdToCategoryId(int elementId)
        {
            return _elementIdToCategoryId[elementId];
        }
        public override string ElementIdToDescription(int elementId)
        {
            return _elementIdToDescription[elementId];
        }
        public override string CategoryIdToDescription(int categoryId)
        {
            if (_categoryIdToDescription == null)
            {
                return categoryId.ToString();
            }
            return _categoryIdToDescription[categoryId];
        }

        private List<Tuple<float, float>> ComputeMeanAndVolatilityForEachChannel()
        {
            const int DistinctValuesToComputeInEachChannel = 3; //sum + sumSquare + count
            var sumSumSquareCountForEachChannel = new float[Channels * DistinctValuesToComputeInEachChannel];
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => UpdateWith_Sum_SumSquare_Count_For_Each_Channel(elementId, sumSumSquareCountForEachChannel, ref nbPerformed));

            var result = new List<Tuple<float, float>>();
            for (int channel = 0; channel < Channels; ++channel)
            {
                var sum = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel];
                var sumSquare = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel + 1];
                var count = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel + 2];
                var mean = (sum / count);
                var variance = (sumSquare / count) - mean * mean;
                var volatility = (float)Math.Sqrt(Math.Max(0, variance));
                _logger.Info("Mean and volatility for channel#" + channel + " : " + mean.ToString(CultureInfo.InvariantCulture) + " ; " + volatility.ToString(CultureInfo.InvariantCulture));
                result.Add(Tuple.Create(mean, volatility));
            }
            return result;
        }
        private void UpdateStatus(ref int nbPerformed)
        {
            int delta = Math.Max(Count / 100, 1);
            var newNbPerformed = Interlocked.Increment(ref nbPerformed);
            if ((newNbPerformed % delta == 0) || (newNbPerformed == Count))
            {
                _logger.Info("Done: " + (100 * newNbPerformed) / Count + "%");
            }
        }
        private void MakeSquarePictures(int elementId, string targetDirectory, bool alwaysUseBiggestSideForWidthSide, bool alwaysCropInsidePicture, Tuple<byte, byte, byte> fillingColor, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            var targetFilenames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
            if (skipIfFileAlreadyExists && AllFileExist(targetFilenames))
            {
                return;
            }
            var bitmapContent = NewBitmapContent(elementId);
            var squarePicture = bitmapContent.MakeSquarePictures(alwaysUseBiggestSideForWidthSide, alwaysCropInsidePicture, fillingColor);
            squarePicture.Save(targetFilenames);
        }
        private static bool AllFileExist(IEnumerable<string> filenames)
        {
            foreach (var filename in filenames)
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
            var targetFilenames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
            if (skipIfFileAlreadyExists && AllFileExist(targetFilenames))
            {
                return;
            }
            var bitmapContent = NewBitmapContent(elementId);
            var cropped = bitmapContent.CropBorder();
            cropped.Save(targetFilenames);
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
        private void UpdateWith_Sum_SumSquare_Count_For_Each_Channel(int elementId, float[] _sum_SumSquare_Count_For_Each_Channel, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            NewBitmapContent(elementId).UpdateWith_Sum_SumSquare_Count_For_Each_Channel(_sum_SumSquare_Count_For_Each_Channel, _ignoreZeroPixel);
        }
        private void Filter(int elementId, string targetDirectory, Func<BitmapContent, bool> isIncluded, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            var targetFilenames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(targetDirectory, subPath)).ToList();
            if (skipIfFileAlreadyExists && AllFileExist(targetFilenames))
            {
                return;
            }
            var bitmapContent = NewBitmapContent(elementId);
            if (isIncluded(bitmapContent))
            {
                bitmapContent.Save(targetFilenames);
            }
        }
        private string StatsFileName => Path.Combine(_dataDirectory, "SharpNet.ini");
        private void CreateStatsFile()
        {
            _logger.Info("Creating stats file");
            var sb = new StringBuilder();
            //We add Mean and volatility stats
            _meanAndVolatilityForEachChannel.Clear();
            sb.Append(MEAN_VOLATILITY + "=");
            foreach (var meanVolatility in ComputeMeanAndVolatilityForEachChannel())
            {
                _meanAndVolatilityForEachChannel.Add(meanVolatility);
                sb.Append(meanVolatility.Item1.ToString(CultureInfo.InvariantCulture) + ";" + meanVolatility.Item2.ToString(CultureInfo.InvariantCulture) + ";");
            }
            sb.Append(Environment.NewLine);
            File.WriteAllText(StatsFileName, sb.ToString());
        }
        const string MEAN_VOLATILITY = "MEAN_VOLATILITY";
        private bool LoadStatsFile()
        {
            _logger.Info("Loading stats file");
            if (!File.Exists(StatsFileName))
            {
                return false;
            }
            bool fileLoadedSuccessfuly = false;
            var content = File.ReadAllLines(StatsFileName);
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
                    fileLoadedSuccessfuly = true;
                }
            }
            _logger.Info(fileLoadedSuccessfuly ? "stats file loaded successfully" : "fail to load stats file");
            return fileLoadedSuccessfuly;
        }
        private double ChannelMean(int channel)
        {
            return _meanAndVolatilityForEachChannel[channel].Item1;
        }
        private double ChannelVolatility(int channel)
        {
            return _meanAndVolatilityForEachChannel[channel].Item2;
        }
        private BitmapContent NewBitmapContent(int elementId)
        {
            var fullNames = _elementIdToSubPath[elementId].Select(subPath => Path.Combine(_dataDirectory, subPath)).ToList();
            if (fullNames.Count == 1 && Channels == 3)
            {
                //single file containing all channels of the element
                return BitmapContent.ValueFomSingleRgbBitmap(fullNames[0], ElementIdToDescription(elementId));
            }
            else
            {
                Debug.Assert(Channels == fullNames.Count);
                //each file contains 1 channel of the element
                return BitmapContent.ValueFromSeveralSingleChannelBitmaps(fullNames, ElementIdToDescription(elementId));
            }
        }
    }
}
