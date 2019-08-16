using SharpNet.CPU;
using SharpNet.Pictures;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SharpNet.Datasets
{
    public class DirectoryDataSetLoader<T> : AbstractDataSetLoader<T> where T : struct
    {
        #region private fields
        private readonly string _csvFileName;
        private readonly string _directoryWithElements;
        private readonly Logger _logger;
        private readonly List<int> _elementIdToCategoryId;
        private readonly string[] _categoryIdToDescription;
        private readonly List<string> _elementIdToFullName = new List<string>();
        private readonly List<Tuple<double, double>> _meanAndVolatlityForEachChannel = new List<Tuple<double, double>>();
        #endregion
        public override int Height { get; }
        public override int Width { get; }
        public override CpuTensor<T> Y { get; }

        public DirectoryDataSetLoader(string csvFileName, string directoryWithElements, Logger logger,
            int channels, int height, int width, string[] categoryIdToDescription)
            : base(channels, categoryIdToDescription.Length)
        {
            Height = height;
            Width = width;

            _csvFileName = csvFileName;
            _logger = logger ?? Logger.ConsoleLogger;
            if (!File.Exists(csvFileName))
            {
                _logger.Info("missing file " + csvFileName);
                throw new ArgumentException("missing file " + csvFileName);
            }

            //we retrieve all files in the directory containing the elements
            _directoryWithElements = directoryWithElements ?? Path.GetDirectoryName(csvFileName) ?? "";
            var fileNameWithoutExtensionToFullPath = new Dictionary<string, string>();
            foreach (var fileInfo in new DirectoryInfo(_directoryWithElements).GetFiles())
            {
                fileNameWithoutExtensionToFullPath[Path.GetFileNameWithoutExtension(fileInfo.FullName)] = fileInfo.FullName;
            }

            var lines = File.ReadAllLines(csvFileName);
            _elementIdToCategoryId = new List<int>();
            for (var index = 0; index < lines.Length; index++)
            {
                var lineContent = lines[index].Split(',', ';').ToArray();
                var description = lineContent[0];
                var categoryIdAsString = lineContent[1];

                if (!int.TryParse(categoryIdAsString, out var categoryId) || categoryId < 0)
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
                var elementFileNameWithoutExtension = Path.GetFileNameWithoutExtension(description) ?? "";
                if (!fileNameWithoutExtensionToFullPath.ContainsKey(elementFileNameWithoutExtension))
                {
                    _logger.Debug("WARNING: no matching file for line: " + lines[index]);
                    continue;
                }
                _elementIdToCategoryId.Add(categoryId);
                //_elementIdToDescription.Add(description);
                _elementIdToFullName.Add(fileNameWithoutExtensionToFullPath[elementFileNameWithoutExtension]);
            }

            _categoryIdToDescription = categoryIdToDescription;

            if (!LoadStatsFile())
            {
                CreateStatsFile();
            }
            //We compute Y 
            Y = CpuTensor<T>.CreateOneHotTensor(ElementIdToCategoryId, _elementIdToCategoryId.Count, Categories);
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<T> buffer)
        {
            var data = BitmapContent.ValueOf(_elementIdToFullName[elementId], "");
            var bufferAsDouble = buffer.Content as double[];
            var bufferAsFloat = buffer.Content as float[];
            for (int channel = 0; channel < data.GetChannels(); ++channel)
            {
                for (int row = 0; row < data.GetHeight(); ++row)
                {
                    for (int col = 0; col < data.GetWidth(); ++col)
                    {
                        var val = (double)data.Get(channel, row, col);
                        // ReSharper disable once RedundantLogicalConditionalExpressionOperand
                        if (IgnoreZeroPixel && data.IzZeroPixel(row, col))
                        {
                            // no normalization
                        }
                        else
                        {
                            val = (val - ChannelMean(channel)) / ChannelVolatility(channel);
                        }
                        var bufferIdx = buffer.Idx(indexInBuffer, channel, row, col);
                        if (bufferAsDouble != null)
                        {
                            bufferAsDouble[bufferIdx] = val;
                        }
                        else
                        {
                            // ReSharper disable once PossibleNullReferenceException
                            bufferAsFloat[bufferIdx] = (float)val;
                        }
                    }
                }
            }
        }
        public DirectoryDataSetLoader<T> CropBorder(bool skipIfFileAlreadyExists = true)
        {
            var targetDirectory = _directoryWithElements + "_cropped";
            if (!Directory.Exists(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            _logger.Info("Cropping " + Count + " elements and copying them in " + targetDirectory);
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => CropBorder(GetFullName(elementId), targetDirectory, skipIfFileAlreadyExists, ref nbPerformed));
            return new DirectoryDataSetLoader<T>(_csvFileName, targetDirectory, _logger, Channels, Height, Width, _categoryIdToDescription);
        }
        public DirectoryDataSetLoader<T> Resize(int newWidth, int newHeight, bool skipIfFileAlreadyExists = true)
        {
            var targetDirectory = _directoryWithElements + "_resize_" + newWidth + "_" + newHeight;
            if (!Directory.Exists(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            _logger.Info("Resizing " + Count + " elements and copying them in " + targetDirectory);
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => Resize(GetFullName(elementId), targetDirectory, newWidth, newHeight, skipIfFileAlreadyExists, ref nbPerformed));
            return new DirectoryDataSetLoader<T>(_csvFileName, targetDirectory, _logger, Channels, Height, Width, _categoryIdToDescription);
        }
        public DirectoryDataSetLoader<T> MakeSquarePictures(bool alwaysUseBiggestSideForWidthSide, Tuple<byte, byte, byte> fillingColor = null, bool skipIfFileAlreadyExists = true)
        {
            fillingColor = fillingColor ?? Tuple.Create((byte)0, (byte)0, (byte)0);
            var targetDirectory = _directoryWithElements + "_square";
            if (!Directory.Exists(targetDirectory))
            {
                Directory.CreateDirectory(targetDirectory);
            }
            _logger.Info("Making " + Count + " elements square pictures and copying them in " + targetDirectory);
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => MakeSquarePictures(GetFullName(elementId), targetDirectory, alwaysUseBiggestSideForWidthSide, fillingColor, skipIfFileAlreadyExists, ref nbPerformed));
            return new DirectoryDataSetLoader<T>(_csvFileName, targetDirectory, _logger, Channels, Height, Width, _categoryIdToDescription);
        }
        public override int Count => _elementIdToCategoryId.Count;
        public override int ElementIdToCategoryId(int elementId)
        {
            return _elementIdToCategoryId[elementId];
        }
        public override string CategoryIdToDescription(int categoryId)
        {
            if (_categoryIdToDescription == null)
            {
                return categoryId.ToString();
            }
            return _categoryIdToDescription[categoryId];
        }

        private List<Tuple<double, double>> ComputeMeanAndVolatilityForEachChannel(bool ignoreZeroPixel)
        {
            int channelCount = 3;
            var _sum_SumSquare_Count_For_Each_Channel = new double[channelCount * 3];
            int nbPerformed = 0;
            Parallel.For(0, Count, elementId => UpdateWith_Sum_SumSquare_Count_For_Each_Channel(GetFullName(elementId), _sum_SumSquare_Count_For_Each_Channel, ignoreZeroPixel, ref nbPerformed));

            var result = new List<Tuple<double, double>>();
            for (int channel = 0; channel < channelCount; ++channel)
            {
                var sum = _sum_SumSquare_Count_For_Each_Channel[3 * channel];
                var sumSquare = _sum_SumSquare_Count_For_Each_Channel[3 * channel + 1];
                var count = _sum_SumSquare_Count_For_Each_Channel[3 * channel + 2];
                var mean = (sum / count);
                var variance = (sumSquare / count) - mean * mean;
                var volatility = Math.Sqrt(Math.Max(0, variance));
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
        private void MakeSquarePictures(string fullName, string targetDirectory, bool alwaysUseBiggestSideForWidthSide, Tuple<byte, byte, byte> fillingColor, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            var targetFilename = Path.Combine(targetDirectory, Path.GetFileName(fullName) ?? "");
            if (skipIfFileAlreadyExists && File.Exists(targetFilename))
            {
                return;
            }
            var bitmapContent = BitmapContent.ValueOf(fullName, "");
            var squarePicture = bitmapContent.MakeSquarePictures(alwaysUseBiggestSideForWidthSide, fillingColor);
            squarePicture.Save(targetFilename);
        }
        private void CropBorder(string fullName, string targetDirectory, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            var targetFilename = Path.Combine(targetDirectory, Path.GetFileName(fullName) ?? "");
            if (skipIfFileAlreadyExists && File.Exists(targetFilename))
            {
                return;
            }
            var bitmapContent = BitmapContent.ValueOf(fullName, "");
            var cropped = bitmapContent.CropBorder();
            cropped.Save(targetFilename);
        }
        private void Resize(string fullName, string targetDirectory, int newWidth, int newHeight, bool skipIfFileAlreadyExists, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            var targetFilename = Path.Combine(targetDirectory, Path.GetFileName(fullName) ?? "");
            if (skipIfFileAlreadyExists && File.Exists(targetFilename))
            {
                return;
            }
            using (var bmp = new Bitmap(fullName))
            {
                var resizedBmp = PictureTools.ResizeImage(bmp, newWidth, newHeight);
                BitmapContent.Save(resizedBmp, targetFilename);
            }
        }
        private void UpdateWith_Sum_SumSquare_Count_For_Each_Channel(string fullName, double[] _sum_SumSquare_Count_For_Each_Channel, bool ignoreZeroPixel, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            BitmapContent.ValueOf(fullName, "").UpdateWith_Sum_SumSquare_Count_For_Each_Channel(_sum_SumSquare_Count_For_Each_Channel, ignoreZeroPixel);
        }
        private const bool IgnoreZeroPixel = true;
        private string StatsFileName => Path.Combine(_directoryWithElements, "SharpNet.ini");
        private void CreateStatsFile()
        {
            _logger.Info("Creating stats file");
            var sb = new StringBuilder();
            //We add Mean and volatility stats
            _meanAndVolatlityForEachChannel.Clear();
            sb.Append(MEAN_VOLATILITY + "=");
            foreach (var meanVolatility in ComputeMeanAndVolatilityForEachChannel(IgnoreZeroPixel))
            {
                _meanAndVolatlityForEachChannel.Add(meanVolatility);
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
                    _meanAndVolatlityForEachChannel.Clear();
                    var values = splitted[1].Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
                    for (int i = 0; i < values.Length; i += 2)
                    {
                        _meanAndVolatlityForEachChannel.Add(Tuple.Create(values[i], values[i + 1]));
                    }

                    fileLoadedSuccessfuly = true;
                }
            }
            _logger.Info(fileLoadedSuccessfuly ? "stats file loaded successfully" : "fail to load stats file");
            return fileLoadedSuccessfuly;
        }
        private double ChannelMean(int channel)
        {
            return _meanAndVolatlityForEachChannel[channel].Item1;
        }
        private double ChannelVolatility(int channel)
        {
            return _meanAndVolatlityForEachChannel[channel].Item2;
        }
        private string GetFullName(int elementId) => _elementIdToFullName[elementId];
    }
}
