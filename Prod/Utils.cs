﻿using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Xml;
using log4net;
using log4net.Config;
using log4net.Util;
using SharpNet.Hyperparameters;
using SharpNet.Pictures;
using Path = System.IO.Path;

namespace SharpNet
{
    public enum Objective_enum
    {
        Regression,
        Classification
    }

    /// <summary>
    /// the loss function (= the objective function)
    /// the goal of the model will be to reduce the value returned by this loss function
    /// (lower is always better)
    /// </summary>
    public enum EvaluationMetricEnum
    {

        /// <summary>
        /// y true : a matrix of shape (batch_size, numClass) with, for each row the 'true' proba of each class
        /// y_predicted: a matrix of shape (batch_size, numClass) with, for each row the predicted proba of each class
        /// works only for metric (to rank submission), do not work as a loss function, higher s better
        /// </summary>
        Accuracy,

        /// <summary>
        /// y true : a sparse matrix of shape (batch_size, 1) with the index of the 'true' class
        /// y_predicted: a matrix of shape (batch_size, numClass) with, for each row the predicted proba of each class
        /// works only for metric (to rank submission), do not work as a loss function, higher s better
        /// </summary>
        SparseAccuracy,

        AccuracyCategoricalCrossentropyWithHierarchy,   // works only for metric (to rank submission), do not work as a loss function, higher s better

        /// <summary>
        /// To be used with sigmoid activation layer.
        /// In a single row, each value will be in [0,1] range
        /// Support of multi labels (one element can belong to several numClass at the same time)
        /// The expected Y value is a binary value: 0 or 1
        /// </summary>
        BinaryCrossentropy, // ok for loss, lower is better

        /// <summary>
        /// To be used with sigmoid activation layer.
        /// In a single row, each value will be in [0,1] range
        /// Support of multi labels (one element can belong to several numClass at the same time)
        /// The expected Y value is a binary value: 0 or 1
        /// </summary>
        BCEWithFocalLoss, // ok for loss, lower is better


        /// <summary>
        /// To be used with sigmoid activation layer.
        /// In a single row, each value will be in [0,1] range
        /// Support of multi labels (one element can belong to several numClass at the same time)
        /// The expected Y value is a continuous value in [0, 1] range (not a binary value: 0 or 1)
        /// </summary>
        BCEContinuousY, // ok for loss, lower is better

        /// <summary>
        /// To be used with softmax activation layer.
        /// In a single row, each value will be in [0,1] range, and the sum of all values wil be equal to 1.0 (= 100%)
        /// Do not support multi labels (each element can belong to exactly 1 category)
        /// </summary>
        CategoricalCrossentropy, // ok for loss, lower is better


        /* Hierarchical Category:
                              Object
                          /           \
                         /             \
                        /               \
                     Fruit             Flower
                      75%                25%
                   /   |   \            |    \
             Cherry  Apple  Orange    Rose    Tulip 
              70%     20%    10%      50%      50%
                     /   \            
                   Fuji  Golden
                    15%   85%
        */
        /// <summary>
        /// To be used with SoftmaxWithHierarchy activation layer.
        /// Each category (parent node) can be divided into several sub categories (children nodes)
        /// For any parent node: all children will have a proba in [0,1] range, and the sum of all children proba will be equal to 1.0 (= 100%)
        /// </summary>
        CategoricalCrossentropyWithHierarchy, // ok for loss, lower is better

        /*
         * Huber loss, see  https://en.wikipedia.org/wiki/Huber_loss
         * */
        Huber, // ok for loss, lower is better

        /*
        * Mean Squared Error loss, see https://en.wikipedia.org/wiki/Mean_squared_error
        * loss = ( predicted - expected ) ^2
        * */
        Mse, // ok for loss, lower is better

        /*
        * Mean Squared Error of log loss,
        * loss = ( log( max(predicted,epsilon) ) - log(expected) ) ^2
        * */
        MseOfLog, // ok for loss, lower is better

        /*
        * Mean Absolute Error loss, see https://en.wikipedia.org/wiki/Mean_absolute_error
        * loss = abs( predicted - expected )
        * */
        Mae, // ok for loss, lower is better

        /*
         * RootMean Squared Error loss, see https://en.wikipedia.org/wiki/Mean_squared_error
         * loss = ( predicted - expected ) ^2
         * */
        Rmse, // ok for loss, lower is better

        F1Micro, // ok for loss, higher is better

        PearsonCorrelation, // works only for metric (to rank submission), do not work as a loss function, higher s better
        SpearmanCorrelation, // works only for metric (to rank submission), do not work as a loss function, higher s better

        //Mean Squared Log Error, see: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error
        //loss = (log(1+predicted) - log(1+expected)) ^ 2
        MeanSquaredLogError, // ok for loss, lower is better

        /// <summary>
        /// To be used with softmax activation layer.
        /// For the prediction:
        ///     In a single row, each value will be in [0,1] range, and the sum of all values wil be equal to 1.0 (= 100%)
        /// For the y_true:
        ///     In a single row, each value will be a scalar integer in the range [0, number_of_categories-1]
        /// Do not support multi labels (each element can belong to exactly 1 category)
        /// </summary>
        SparseCategoricalCrossentropy, // ok for loss, lower is better


        //Area Under the Curve, see: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
        AUC, // works only for metric (to rank submission), do not work as a loss function, higher s better

        //Average Precision Score, see : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        AveragePrecisionScore, // works only for metric (to rank submission), do not work as a loss function, higher s better


        DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE, // default value, do not use
    }


    public static class Utils
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(Utils));


        public static int[] CloneShapeWithNewCount(int[] shape, int newCount)
        {
            if (shape == null)
            {
                return null;
            }
            var result = (int[])shape.Clone();
            result[0] = newCount;
            return result;
        }
        public static long LongProduct(int[] data)
        {
            return LongProduct(data.Select(i=>(long)i).ToArray());
        }

        private static long LongProduct(long[] data)
        {
            if ((data == null) || (data.Length == 0))
            {
                return 0;
            }

            long result = data[0];
            for (int i = 1; i < data.Length; ++i)
            {
                result *= data[i];
            }

            return result;
        }
        public static int Product(int[] data)
        {
            if ((data == null) || (data.Length == 0))
            {
                return 0;
            }

            var result = data[0];
            for (int i = 1; i < data.Length; ++i)
            {
                result *= data[i];
            }

            return result;
        }
        // ReSharper disable once UnusedMember.Global
        public static ulong AvailableRamMemoryInBytes()
        {
            var ramCounter = new PerformanceCounter("Memory", "Available Bytes");
            return (ulong)ramCounter.NextValue();
        }
        public static IList<IList<T>> AllPermutations<T>(List<T> data)
        {
            var result = new List<IList<T>>();
            AllPermutationsHelper(data, 0, result);
            return result;
        }

        public static string ToString(EvaluationMetricEnum evaluationMetric)
        {
            switch (evaluationMetric)
            {
                case EvaluationMetricEnum.SparseAccuracy:
                    return ToString(EvaluationMetricEnum.Accuracy);
                case EvaluationMetricEnum.SparseCategoricalCrossentropy:
                    return ToString(EvaluationMetricEnum.CategoricalCrossentropy);
                default:
                    return evaluationMetric.ToString();
            }
        }

        public static bool HigherScoreIsBetter(EvaluationMetricEnum evaluationMetric)
        {
            switch (evaluationMetric)
            {
                case EvaluationMetricEnum.Accuracy:
                case EvaluationMetricEnum.SparseAccuracy:
                case EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy:
                case EvaluationMetricEnum.F1Micro:
                case EvaluationMetricEnum.PearsonCorrelation:
                case EvaluationMetricEnum.SpearmanCorrelation:
                case EvaluationMetricEnum.AUC:
                case EvaluationMetricEnum.AveragePrecisionScore:
                    return true; // higher is better
                case EvaluationMetricEnum.BinaryCrossentropy:
                case EvaluationMetricEnum.BCEContinuousY:
                case EvaluationMetricEnum.BCEWithFocalLoss:
                case EvaluationMetricEnum.CategoricalCrossentropy:
                case EvaluationMetricEnum.SparseCategoricalCrossentropy:
                case EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy:
                case EvaluationMetricEnum.Huber:
                case EvaluationMetricEnum.Mae:
                case EvaluationMetricEnum.Mse:
                case EvaluationMetricEnum.MseOfLog:
                case EvaluationMetricEnum.MeanSquaredLogError:
                case EvaluationMetricEnum.Rmse:
                    return false; // lower is better
                default:
                    throw new NotImplementedException($"unknown {nameof(EvaluationMetricEnum)} : {evaluationMetric}");
            }
        }
        /// <summary>
        /// true if score 'a' is better then score 'b'
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="metricEnum"></param>
        /// <returns></returns>
        public static bool IsBetterScore(float a, float b, EvaluationMetricEnum metricEnum)
        {
            if (float.IsNaN(a))
            {
                return false;
            }
            if (float.IsNaN(b))
            {
                return true;
            }
            if (HigherScoreIsBetter(metricEnum))
            {
                return a > b;
            }
            else
            {
                return a < b;
            }
        }
        /// <summary>
        /// duplicate the input list 'data' by 'repeatCount' time:
        /// Each element of the initial list will be duplicated 'repeatCount' time:
        /// Ex:
        /// if
        ///     data = [1,2 3, 2] and repeatCount = 2
        /// then
        ///     output = [1,1, 2,2, 3,3, 2,2]
        /// </summary>
        /// <returns></returns>
        public static List<T> Repeat<T>(IEnumerable<T> data, int repeatCount)
        {
            var result = new List<T>();
            foreach (var t in data)
            {
                for (int i = 0; i < repeatCount; ++i)
                {
                    result.Add(t);
                }
            }
            return result;
        }
        public static string ShapeToStringWithBatchSize(int[] shape)
        {
            if (shape == null)
            {
                return "(?)";
            }

            return "(None, " + string.Join(", ", shape.Skip(1)) + ")";
        }
        public static string ShapeToString(int[] shape)
        {
            if (shape == null)
            {
                return "(?)";
            }

            return "(" + string.Join(", ", shape) + ")";
        }
        public static ulong Sum(this IEnumerable<ulong> vector)
        {
            ulong result = 0;
            foreach (var d in vector)
            {
                result += d;
            }

            return result;
        }
        public static string MemoryBytesToString(ulong bytes)
        {
            if (bytes > 15_000_000_000)
            {
                return (bytes / 1_000_000_000) + "GB";
            }

            if (bytes > 3_000_000)
            {
                return (bytes / 1_000_000) + "MB";
            }

            if (bytes > 3_000)
            {
                return (bytes / 1_000) + "KB";
            }

            return bytes + "B";
        }
        public static double Interpolate(List<Tuple<double, double>> values, double x, bool constantByInterval = false)
        {
            if (values.Count == 1)
            {
                return values[0].Item2;
            }

            for (int i = 0; i < values.Count; ++i)
            {
                var x2 = values[i].Item1;
                if (x > x2)
                {
                    continue;
                }

                var y2 = values[i].Item2;
                if ((Math.Abs(x2 - x) < 1e-9) || i == 0)
                {
                    return y2;
                }

                Debug.Assert(x < x2);
                var x1 = values[i - 1].Item1;
                Debug.Assert(x > x1);
                var y1 = values[i - 1].Item2;
                if (constantByInterval)
                {
                    return y1;
                }

                return Interpolate(x1, y1, x2, y2, x);
            }

            return values.Last().Item2;
        }
        public static double Interpolate(double x1, double y1, double x2, double y2, double xToInterpolate)
        {
            double dEpoch = (xToInterpolate - x1) / (x2 - x1);
            double deltaLearningRate = (y2 - y1);
            return y1 + dEpoch * deltaLearningRate;
        }
        public static void UniformDistribution(Span<float> toRandomize, Random rand, double minValue, double maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (float)(minValue + rand.NextDouble() * (maxValue - minValue));
            }
        }
        public static void UniformDistribution(Span<byte> toRandomize, Random rand, byte minValue, byte maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (byte)(minValue + rand.Next(maxValue - minValue + 1));
            }
        }
        public static void NormalDistribution(Span<float> toRandomize, Random rand, double mean, double stdDev)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (float)NextDoubleNormalDistribution(rand, mean, stdDev);
            }
        }
        public static void UniformDistribution(Span<int> toRandomize, Random rand, int minValue, int maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = rand.Next(minValue, maxValue+1);
            }
        }
        /// <summary>
        /// compute the mean and volatility of 'data'
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        // ReSharper disable once UnusedMember.Global
        public static (float mean, float volatility) MeanAndVolatility(ReadOnlySpan<float> data)
        {
            if (data.Length == 0)
            {
                return (0f, 0f);
            }

            double sum = 0f;
            double sumSquare = 0.0;
            foreach (var val in data)
            {
                sum += val;
                sumSquare += val * val;
            }

            var mean = (sum / data.Length);
            var variance = (sumSquare / data.Length) - mean * mean;
            var volatility = Math.Sqrt(Math.Max(0, variance));
            return ((float)mean, (float)volatility);
        }
        public static void Shuffle<T>(IList<T> list, Random rand)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rand.Next(n + 1);
                (list[k], list[n]) = (list[n], list[k]);
            }
        }
        public static void Shuffle<T>(IList<T> list, Random rand, int blockSize)
        {
            Debug.Assert(list.Count%blockSize == 0);
            var blockIds =Enumerable.Range(0, list.Count / blockSize).ToList();
            Shuffle(blockIds, rand);
            var listCopy = new List<T>(list);
            foreach (var t in blockIds)
            {
                for (int j = 0; j < blockSize; ++j)
                {
                    list[t*blockSize + j] = listCopy[t*blockSize + j];
                }
            }
        }
        public static int FirstMultipleOfAtomicValueAboveOrEqualToMinimum(int minimum, int atomicValue)
        {
            if (minimum % atomicValue != 0)
            {
                minimum += atomicValue - minimum % atomicValue;
            }

            return minimum;
        }
        public static string UpdateFilePathChangingExtension(string filePath, string prefix, string suffix,
            string newExtension)
        {
            string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(filePath);
            if (!newExtension.StartsWith("."))
            {
                newExtension = "." + newExtension;
            }

            string path = GetDirectoryName(filePath);
            return ConcatenatePathWithFileName(path, prefix + fileNameWithoutExtension + suffix + newExtension);
        }

        public static void WriteBinaryFile<T>(string fileName, T[] values) where T : struct
        {
            using var stream = File.Open(fileName, FileMode.Create);
            using var writer = new BinaryWriter(stream, Encoding.UTF8, false);
            var bytes = MemoryMarshal.Cast<T, byte>(values);
            writer.Write(bytes);
        }


        /// <summary>
        /// read a part of a binary file, starting at position 'startIndex' in the file
        /// </summary>
        /// <param name="fileName">file to read</param>
        /// <param name="startIndex">the fistIndex to read in the file</param>
        /// <param name="arrayLength">number of elements to read</param>
        /// <returns>an array of 'arrayLength' elements of type 'T'</returns>
        public static T[] ReadArrayFromBinaryFile<T>(string fileName, int startIndex, int arrayLength) where T : struct
        {
            var res = new T[arrayLength];
            LoadBufferFromBinaryFile(fileName, startIndex, res.AsSpan());
            return res;
        }

        private static void LoadBufferFromBinaryFile<T>(string fileName, int startIndex, Span<T> buffer) where T : struct
        {
            var bytesSpan = MemoryMarshal.Cast<T, byte>(buffer);
            int tSize = Marshal.SizeOf(typeof(T));
            // Open file with a BinaryReader
            using var b = new BinaryReader(File.Open(fileName, FileMode.Open, FileAccess.Read, FileShare.Read));
            // Seek to our required position 'startIndex'
            b.BaseStream.Seek(startIndex * tSize, SeekOrigin.Begin);
            // ReSharper disable once MustUseReturnValue
            b.Read(bytesSpan);
        }
        public static string ConcatenatePathWithFileName(string path, params string[] subPaths)
        {
            string result = path;
            foreach (var t in subPaths)
            {
                result = Path.Combine(result, t);
            }

            return result;
        }
        //private static readonly Dictionary<string, List<string[]>> ReadCsvCache = new();
        //private static readonly object LockObject = new();
        //public static List<string[]> ReadCsvWithCache(string csvPath, char? mandatorySeparator = null)
        //{
        //    lock (LockObject)
        //    {
        //        if (!ReadCsvCache.ContainsKey(csvPath))
        //        {
        //            ReadCsvCache[csvPath] = ReadCsv(csvPath, mandatorySeparator).ToList();
        //        }
        //        return ReadCsvCache[csvPath];
        //    }
        //}
        /// <summary>
        /// Read all rows of a CSV file
        /// if the separator (parameter: mandatorySeparator) is not provided, it will be detected automatically
        /// </summary>
        /// <param name="csvPath"></param>
        /// <param name="mandatorySeparator">the separator to use (if provided)
        /// if it is not provided, the CSV separator will be detected automatically (preferred method)
        /// </param>
        /// <returns></returns>
        public static IEnumerable<string[]> ReadCsv(string csvPath, char? mandatorySeparator = null)
        {
            using TextReader fileReader = File.OpenText(csvPath);
            var csvConfig = new CsvHelper.Configuration.CsvConfiguration(CultureInfo.InvariantCulture)
            {
                TrimOptions = CsvHelper.Configuration.TrimOptions.InsideQuotes | CsvHelper.Configuration.TrimOptions.Trim,
                BadDataFound = null,
            };
            if (mandatorySeparator.HasValue)
            {
                csvConfig.DetectDelimiter = false;
                csvConfig.Delimiter = mandatorySeparator.Value.ToString();
            }
            else
            {
                csvConfig.DetectDelimiter = true;
            }

            var csvParser = new CsvHelper.CsvParser(fileReader, csvConfig);

            while (csvParser.Read())
            {
                string[] row = csvParser.Record;
                if (row == null)
                {
                    break;
                }
                yield return row;
            }
        }

        //!D TODO Add tests
        public static float TryParseFloat(ReadOnlySpan<char> lineSpan, int nextItemStart, int nextItemLength)
        {
            const float invalid_float = float.NaN;
            switch (nextItemLength)
            {
                case <= 0: return invalid_float;
                case 1: return char.IsDigit(lineSpan[nextItemStart]) ? (lineSpan[nextItemStart] - '0') : invalid_float;
                default: return float.TryParse(lineSpan.Slice(nextItemStart, nextItemLength), out var floatValue) ? floatValue : invalid_float;
            }
        }

        //!D TODO Add tests
        public static int TryParseInt(ReadOnlySpan<char> lineSpan, int nextItemStart, int nextItemLength)
        {
            const int invalid_int = 0; //TODO: return something more specific
            switch (nextItemLength)
            {
                case <= 0: return invalid_int;
                case 1: return char.IsDigit(lineSpan[nextItemStart]) ? (lineSpan[nextItemStart] - '0') : invalid_int;
                default: return int.TryParse(lineSpan.Slice(nextItemStart, nextItemLength), out var intValue) ? intValue : invalid_int;
            }
        }

        public static string SubStringWithCache(ReadOnlySpan<char> lineSpan, int nextItemStart, int nextItemLength, ConcurrentDictionary<int, string> cache)
        {
            var strSpan = lineSpan.Slice(nextItemStart, nextItemLength);
            var hashStrSpan = string.GetHashCode(strSpan);
            if (cache.TryGetValue(hashStrSpan, out var str) && strSpan.Equals(str, StringComparison.Ordinal))
            {
                return str;
            }
            //we need to allocate the string
            str = strSpan.ToString();
            cache.TryAdd(hashStrSpan, str);
            return str;
        }

        public static string NormalizeCategoricalFeatureValue(string value)
        {
            if (!value.Any(CharToBeRemovedInStartOrEnd))
            {
                return value;
            }

            var sb = new StringBuilder(value.Length);
            int currentContinuousSpaces = 0;
            foreach (var c in value)
            {
                if (!CharToBeRemovedInStartOrEnd(c))
                {
                    currentContinuousSpaces = 0;
                    sb.Append(c);
                }
                else
                {
                    if (sb.Length != 0)
                    {
                        sb.Append(' ');
                        ++currentContinuousSpaces;
                    }
                }
            }
            if (currentContinuousSpaces != 0)
            {
                sb.Remove(sb.Length - currentContinuousSpaces, currentContinuousSpaces);
            }
            return sb.ToString();
        }

        private static bool CharToBeRemovedInStartOrEnd(char c)
        {
            return char.IsWhiteSpace(c) ||c == '\"' || c == '\n' || c == '\r' || c == ';' || c == ',';
        }

        /// <summary>
        /// return the intersection of list a and b
        /// (elements that are in both 'a' and 'b')
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static List<T> Intersect<T>(IList<T> a, IList<T> b)
        {
            var result = new List<T>();
            if (a == null || b == null ||a.Count == 0 || b.Count == 0)
            {
                return result;
            }
            var bHash = new HashSet<T>(b);

            foreach (var e in a)
            {
                if (bHash.Contains(e))
                {
                    result.Add(e);
                }
            }
            return result;
        }

        public static List<T> Without<T>(IEnumerable<T> a, T b)
        {
            return Without(a, new List<T> { b });
        }

        public static List<T> Without<T>(IEnumerable<T> a, IEnumerable<T> b)
        {
            var result = new List<T>();
            if (a == null || b == null)
            {
                return result;
            }
            var bHash = new HashSet<T>(b);

            foreach (var aItem in a)
            {
                if (!bHash.Contains(aItem))
                {
                    result.Add(aItem);
                }
            }
            return result;
        }
        // ReSharper disable once UnusedMember.Global
        public static List<T> Join<T>(IEnumerable<T> a, IEnumerable<T> b)
        {
            var result = new List<T>();
            if (a == null)
            {
                return b == null ? result : b.ToList();
            }

            if (b == null)
            {
                return a.ToList();
            }

            var first = a.ToList();
            first.AddRange(b);
            return first;
        }
        public static long FileLength(string path)
        {
            return new FileInfo(path).Length;
        }
        
        public static bool TryGet<T>(this IDictionary<string, object> serialized, string key, out T value)
        {
            if (serialized.TryGetValue(key, out var resAsObject))
            {
                value = (T)resAsObject;
                return true;
            }

            value = default;
            return false;
        }
        public static T GetOrDefault<T>(this IDictionary<string, object> serialized, string key, T defaultValue)
        {
            if (serialized.TryGetValue(key, out var resAsObject))
            {
                return (T)resAsObject;
            }
            return defaultValue;
        }

        // ReSharper disable once UnusedMember.Global
        public static T TryGet<T>(this IDictionary<string, object> serialized, string key)
        {
            if (serialized.TryGetValue(key, out var resAsObject))
            {
                return (T)resAsObject;
            }

            return default;
        }
        //public static bool Equals<T>(T a, T b, string id, ref string errors)
        //{
        //    if (!Equals(a, b))
        //    {
        //        errors += id + ": " + a + " != " + b + Environment.NewLine;
        //        return false;
        //    }

        //    return true;
        //}
        public static bool Equals(double a, double b, double epsilon, string id, ref string errors)
        {
            if (Math.Abs(a - b) > epsilon)
            {
                errors += id + ": " + a + " != " + b + Environment.NewLine;
                return false;
            }

            return true;
        }
        public static double BetaDistribution(double a, double b, Random rand)
        {
            var alpha = a + b;
            double beta;
            if (Math.Min(a, b) <= 1.0)
            {
                beta = Math.Max(1 / a, 1 / b);
            }
            else
            {
                beta = Math.Sqrt(alpha - 2.0) / (2 * a * b - alpha);
            }

            double gamma = a + 1 / beta;
            double w;
            while (true)
            {
                var u1 = rand.NextDouble();
                var u2 = rand.NextDouble();
                var v = beta * Math.Log(u1 / (1 - u1));
                w = a * Math.Exp(v);
                var tmp = Math.Log(alpha / (b + w));
                if ((alpha * tmp + (gamma * v) - 1.3862944) >= Math.Log(u1 * u1 * u2))
                {
                    break;
                }
            }

            return w / (b + w);
        }
        public static string LoadResourceContent(Assembly assembly, string resourceName)
        {
            using (var resourceStream = assembly.GetManifestResourceStream(resourceName))
                // ReSharper disable once AssignNullToNotNullAttribute
            using (var reader = new StreamReader(resourceStream, Encoding.UTF8))
            {
                return reader.ReadToEnd();
            }
        }
        public static T2[] Select<T1, T2>(this ReadOnlySpan<T1> s, Func<T1, T2> select)
        {
            var res = new T2[s.Length];
            for (int i = 0; i < s.Length; ++i)
            {
                res[i] = select(s[i]);
            }

            return res;
        }
        public static int Count<T>(this ReadOnlySpan<T> s, Func<T, bool> isIncluded)
        {
            int result = 0;
            foreach (var t in s)
            {
                if (isIncluded(t))
                {
                    ++result;
                }
            }

            return result;
        }
        public static float Max(this ReadOnlySpan<float> s)
        {
            var result = float.MinValue;
            foreach (var t in s)
            {
                result = Math.Max(result, t);
            }

            return result;
        }
        public static float Min(this ReadOnlySpan<float> s)
        {
            var result = float.MaxValue;
            foreach (var t in s)
            {
                result = Math.Min(result, t);
            }

            return result;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float x)
        {
            return (float)(1 / (1 + Math.Exp(-x)));
        }
        public static string GetString(XmlNode node, string keyName)
        {
            return node?.SelectSingleNode(keyName)?.InnerText ?? "";
        }
        public static int GetInt(XmlNode node, string keyName, int defaultValue)
        {
            return int.TryParse(GetString(node, keyName), out var result) ? result : defaultValue;
        }
        public static bool GetBool(XmlNode node, string keyName, bool defaultValue)
        {
            return bool.TryParse(GetString(node, keyName), out var result) ? result : defaultValue;
        }
        public static void ConfigureGlobalLog4netProperties(string logDirectory, string logFile, bool overwriteIfExists = true)
        {
            lock (lockConfigureLog4netProperties)
            {
                ConfigureLog4netProperties(logDirectory, logFile, GlobalContext.Properties, overwriteIfExists);
                XmlConfigurator.Configure(LogManager.GetRepository(Assembly.GetEntryAssembly()), new FileInfo(@"log4net.config"));
            }
        }
        public static void ConfigureThreadLog4netProperties(string logDirectory, string logFile, bool overwriteIfExists = true)
        {
            lock (lockConfigureLog4netProperties)
            {
                ConfigureLog4netProperties(logDirectory, logFile, ThreadContext.Properties, overwriteIfExists);
                XmlConfigurator.Configure(LogManager.GetRepository(Assembly.GetEntryAssembly()), new FileInfo(@"log4net.config"));
            }
        }
        public static void ConfigureThreadIdLog4netProperties()
        {
            lock (lockConfigureLog4netProperties)
            {
                ThreadContext.Properties["threadid"] = Thread.CurrentThread.ManagedThreadId;
            }
        }
        /// <summary>
        /// return the SHA-1 of the image file (160 bits stored in a string of 40 bytes in hexadecimal format: 0=>f)
        /// ignoring all metadata associated with the image
        /// </summary>
        /// <param name="imagePath">path to the image</param>
        /// <returns>
        /// empty string if the file do not exists
        /// the SHA-1 of the file if it exists
        /// </returns>
        public static string ImagePathToSHA1(string imagePath)
        {
            try
            {
                return BitmapContent.ValueFomSingleRgbBitmap(imagePath).SHA1();
            }
            catch (Exception e)
            {
                Log.Error("error", e);
                return "";
            }
        }
        /// <summary>
        /// return the SHA-1 of a file (160 bits stored in a string of 40 bytes in hexadecimal format: 0=>f)
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns>
        /// empty string if the file do not exists
        /// the SHA-1 of the file if it exists
        /// </returns>
        // ReSharper disable once UnusedMember.Global
        public static string FileSHA1(string filePath)
        {
            if (!File.Exists(filePath))
            {
                return "";
            }

            using var fs = new FileStream(filePath, FileMode.Open);
            using var bs = new BufferedStream(fs);
#pragma warning disable SYSLIB0021
            using var sha1 = new SHA1Managed();
#pragma warning restore SYSLIB0021
            var hash = sha1.ComputeHash(bs);
            var formatted = new StringBuilder(2 * hash.Length);
            foreach (byte b in hash)
            {
                formatted.AppendFormat("{0:X2}", b);
            }

            return formatted.ToString();
        }
        public static int NearestInt(double d)
        {
            return (int)Math.Round(d);
        }

        public static int PrevPowerOf2(int n)
        {
            if (n < 1)
            {
                return 0;
            }

            var result = 1;
            while (2*result <= n)
            {
                result *=2;
            }

            return result;
        }

        public static int NextPowerOf2(int n)
        {
            if (n == 0)
            {
                return 1;
            }

            n--;
            n |= n >> 1; // Divide by 2^k for consecutive doublings of k up to 32,
            n |= n >> 2; // and then or the results.
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            n++; // The result is a number of 1 bits equal to the number
            // of bits in the original number, plus 1. That's the
            // next highest power of 2.
            return n;
        }

        public static bool SameContent(float[] a, float[] b, double epsilon)
        {
            return SameContent(a, b, epsilon, out _);
        }

        public static bool SameContent(Half[] a, float[] b, double epsilon, out string difference)
        {
            return SameContent(a.Select(h => (float)h).ToArray(), b, epsilon, out difference);
        }

        public static bool SameContent(float[] a, float[] b, double epsilon, out string difference)
        {
            difference = "";
            if (a.Length != b.Length)
            {
                difference = $"different length: {a.Length} vs {b.Length}";
                return false;
            }

            for (int i = 0; i < a.Length; ++i)
            {
                if (double.IsNaN(a[i]) != double.IsNaN(b[i]))
                {
                    difference = $"different NaN at index {i}: {a[i]} vs {b[i]}";
                    return false;
                }

                if (Math.Abs(a[i] - b[i]) > epsilon)
                {
                    difference = $"different value at index {i}: {a[i]} vs {b[i]}";
                    return false;
                }
            }

            return true;
        }
        /// <summary>
        /// </summary>
        /// <param name="version"></param>
        /// <returns></returns>
        public static Version NewVersion(int version)
        {
            if (version > 100000)
            {
                // version contains(10000 major + 100 minor + build).
                // For example, 12.1.5 would be represented by 120105
                return NewVersionXXYYZZ(version);
            }
            //  version contains(1000 major + 100 minor + build).
            // For example, 7.6.5 would be represented by 7605
            return new Version(version / 1000, (version / 100) % 10, version % 100);
        }
        /// <summary>
        ///  version contains(1000 major + 10 minor).
        /// For example, 9.2 would be represented by 9020
        /// </summary>
        /// <param name="version"></param>
        /// <returns></returns>
        public static Version NewVersionXXYY0(int version)
        {
            var major = version / 1000;
            var minor = (version % 1000) / 10;
            return new Version(major, minor);
        }

        public static Version NewVersionXXYYZZ(int version)
        {
            return new Version(version / 10000, (version / 100) % 100, version % 100);
        }


        public static string ComputeHash(string input, int maxLength)
        {
            // Use input string to calculate MD5 hash
            var sb = new StringBuilder();
            using MD5 md5 = MD5.Create();
            var inputBytes = Encoding.ASCII.GetBytes(input);
            var hashBytes = md5.ComputeHash(inputBytes);

            // Convert the byte array to hexadecimal string
            foreach (var t in hashBytes)
            {
                sb.Append(t.ToString("X2"));
            }

            return sb.ToString().Substring(0, maxLength);
        }

        private static int? _cacheCoreCount;
        public static int CoreCount
        {
            get
            {
                if (!_cacheCoreCount.HasValue)
                {
                    int coreCount = 0;
                    foreach (var item in new System.Management.ManagementObjectSearcher("Select * from Win32_Processor").Get())
                    {
                        coreCount += int.Parse(item["NumberOfCores"].ToString() ?? "");
                    }
                    _cacheCoreCount = coreCount;
                }

                return _cacheCoreCount.Value;

            }
        }
        /// <summary>
        /// Compute the % of time to invest on each use case, knowing the error associated with each use case
        /// </summary>
        /// <param name="errors">
        /// each use case is a tuple with 3 values:
        ///     Item1 : the error to minimize
        ///     Item2 : the volatility around this error
        ///     Item3 : the number of experiments made to compute this error
        /// </param>
        /// <returns>
        /// For each use case, the % of time (between 0 and 1.0) we are willing to invest to explore this use case
        ///  => a value close to 1 means we want to invest most of our time on this use case (because it seems very promising
        ///  => a value close to 0 means we want to invest very little time on this use case (because it doesn't seem use full)
        /// </returns>
        public static double[] TargetCpuInvestmentTime(List<Tuple<double, double, long>> errors)
        {
            double[] result = new double[errors.Count];
            //by default we want to invest the exact same time for each parameter
            for (int i = 0; i < errors.Count; ++i)
            {
                result[i] = 1.0 / errors.Count;
            }
            if (errors.Count <= 1)
            {
                return result;
            }

            var valueWithIndex = new List<Tuple<Tuple<double, double, long>, int>>();
            for (int i = 0; i < errors.Count; ++i)
            {
                if (errors[i].Item3 >= 3)
                {
                    //if the cost is computed on at least 3 samples, we can rely on this cost
                    valueWithIndex.Add(Tuple.Create(errors[i], i));
                }
            }
            if (valueWithIndex.Count <= 1)
            {
                //we have 1 (or 0) use case with relevant info : we'll use the same amount of time for all use cases
                return result; 
            }

            //we order all relevant use cases (at least 2 experiments) from the lowest to the max error
            valueWithIndex = valueWithIndex.OrderBy((t => t.Item1.Item1)).ToList();
            var weights = new List<double>();
            weights.Add(1);
            var bestUseCase = valueWithIndex[0].Item1;
            var lowestError = bestUseCase.Item1;
            var volatilityOfBestUseCase = bestUseCase.Item2;
            const double minWeight = 0.1;
            for (int i = 1; i < valueWithIndex.Count; ++i)
            {
                var currentUseCase = valueWithIndex[i].Item1;
                var currentError = currentUseCase.Item1;
                var currentVolatility = currentUseCase.Item2;
                var volatility = Math.Max(volatilityOfBestUseCase, currentVolatility);

                var lowestErrorInfMargin = lowestError - volatility;
                var lowestErrorSupMargin = lowestError + volatility;
                var currentErrorInfMargin = currentError - volatility;

                if (currentErrorInfMargin >= lowestErrorSupMargin)
                {
                    weights.Add(minWeight);
                    continue;
                }

                double percentageInCommon = (lowestErrorSupMargin - currentErrorInfMargin) /  (lowestErrorSupMargin - lowestErrorInfMargin);
                double weight = percentageInCommon / (2 - percentageInCommon);
                Debug.Assert(weight<=1.0001);
                Debug.Assert(weight>=0.0);
                weights.Add(Math.Max(minWeight, weight));
            }

            double expectedWeightSum = valueWithIndex.Count / ((double)errors.Count);
            double observedWeightSum = weights.Sum();

            for (var i = 0; i < valueWithIndex.Count; i++)
            {
                var normalizedWeights = weights[i]*(expectedWeightSum/ observedWeightSum);
                result[valueWithIndex[i].Item2 ] = normalizedWeights;
            }
            Debug.Assert(Math.Abs(result.ToList().Sum()-1)<=1e-5);
            return result;
        }
        public static int RandomIndexBasedOnWeights(double[] weights, Random rand)
        {
            if (weights.Length <= 1)
            {
                return 0;
            }
            Debug.Assert(weights.Min() >= 0.0);
            var targetSum = weights.Sum() * rand.NextDouble();
            var currentSum = 0.0;
            for (int i = 0; i < weights.Length; i++)
            {
                currentSum += weights[i];
                if (targetSum <= currentSum)
                {
                    return i;
                }
            }
            return weights.Length - 1;
        }
        public static string FieldValueToJsonString(object fieldValue)
        {
            if (fieldValue == null)
            {
                return "";
            }

            if (fieldValue is IList)
            {
                List<string> elements = new();
                foreach (var o in (IList)fieldValue)
                {
                    elements.Add(FieldValueToJsonString(o));
                }
                return "["+string.Join(",", elements)+"]";
            }
            if (fieldValue is bool)
            {
                // ReSharper disable once PossibleNullReferenceException
                return fieldValue.ToString().ToLower();
            }

            var asString = FieldValueToString(fieldValue);
            if (fieldValue is string || fieldValue.GetType().IsEnum)
            {
                asString = "\""+asString+"\"";
            }
            return asString;
        }
        public static string FieldValueToString(object fieldValue)
        {
            if (fieldValue == null)
            {
                return "";
            }
            if (fieldValue is string)
            {
                return (string)fieldValue;
            }
            if (fieldValue is bool ||  fieldValue is int)
            {
                return fieldValue.ToString();
            }
            if (fieldValue is float)
            {
                return ((float)fieldValue).ToString(CultureInfo.InvariantCulture);
            }
            if (fieldValue is double)
            {
                return ((double)fieldValue).ToString(CultureInfo.InvariantCulture);
            }
            if (fieldValue.GetType().IsEnum)
            {
                return fieldValue.ToString();
            }

            if (fieldValue is IList)
            {
                List<string> elements = new();
                foreach (var o in (IList)fieldValue)
                {
                    elements.Add(FieldValueToString(o));
                }
                return string.Join(",", elements);
            }

            throw new ArgumentException($"can transform to string field {fieldValue} of type {fieldValue.GetType()}");
        }
        public static IDictionary<string, object> FromString2String_to_String2Object(IDictionary<string, string> dicoString2String)
        {
            var dicoString2Object = new Dictionary<string, object>();
            foreach (var (key, value) in dicoString2String)
            {
                dicoString2Object[key] = value;
            }
            return dicoString2Object;
        }
        public static void TryDelete(IEnumerable<string> filePaths)
        {
            foreach(var filePath in filePaths)
            {
                TryDelete(filePath);
            }  
        }
       public static bool TryDelete(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
            {
                return false;
            }
            if (!File.Exists(filePath))
            {
                return false;
            }
            try
            {
                File.Delete(filePath);
                return true;
            }
            catch
            {
                return false;
            }
        }
        public static string ChallengesPath => @"C:\Projects\Challenges";
        public static List<string> Launch(string workingDirectory, string exePath, string arguments, ILog log, bool returnOutputedLines)
        {
            var outputLines = returnOutputedLines?new List<string>():null;
            Log.Debug($"Launching {exePath} {arguments} with WorkingDirectory={workingDirectory}");
            var errorDataReceived = "";
            var engineName = Path.GetFileNameWithoutExtension(exePath);
            var psi = new ProcessStartInfo(exePath)
            {
                WorkingDirectory = workingDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                Arguments = arguments,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };
            var process = Process.Start(psi);
            if (process == null)
            {
                string errorMsg = "Fail to start " + engineName + " Engine";
                log.Fatal(errorMsg);
                throw new Exception(errorMsg);
            }
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data != null)
                {
                    errorDataReceived = e.Data;
                }
            };
            process.OutputDataReceived += (_, e) =>
            {
                outputLines?.Add(e.Data);
                if (string.IsNullOrEmpty(e.Data)
                    || e.Data.Contains("Object info sizes") 
                    || e.Data.Contains("Skipping test eval output") 
                    || e.Data.Contains(" min passed")
                    || e.Data.Contains("No further splits with positive gain")
                    || e.Data.Contains("remaining:")
                    || e.Data.Contains("seconds elapsed")
                    || e.Data.Contains("[Info] Iteration:")
                   )
                {
                    return;
                }
                log.Debug(e.Data);
            };
            process.BeginErrorReadLine();
            process.BeginOutputReadLine();
            process.WaitForExit();
            if (!string.IsNullOrEmpty(errorDataReceived) || process.ExitCode != 0)
            {
                if (!(errorDataReceived??"").Contains("is not implemented on GPU"))
                {
                    var errorMsg = "Error in " + engineName + " " + errorDataReceived;
                    log.Fatal(errorMsg);
                    throw new Exception(errorMsg);
                }
            }
            return outputLines;
        }
        private static void AllPermutationsHelper<T>(List<T> data, int i, IList<IList<T>> result)
        {
            if (i == data.Count - 1)
            {
                result.Add(new List<T>(data));
                return;
            }

            //var alreadyUsed = new HashSet<T>(); //to discard duplicate solutions
            for (var j = i; j < data.Count; ++j)
            {
                //if (!alreadyUsed.Add(data[j])) continue; //to discard duplicate solutions
                var tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
                AllPermutationsHelper(data, i + 1, result);
                tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
            }
        }
        private static double NextDoubleNormalDistribution(Random rand, double mean, double stdDev)
        {
            //uniform(0,1) random double
            var u1 = rand.NextDouble();
            //uniform(0,1) random double
            var u2 = rand.NextDouble();
            //random normal(0,1)
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            //random normal(mean,stdDev^2)
            return mean + stdDev * randStdNormal;
        }
        private static string GetDirectoryName(string path)
        {
            try
            {
                if (string.IsNullOrEmpty(path))
                {
                    return "";
                }

                return Path.GetDirectoryName(path);
            }
            catch (Exception)
            {
                return "";
            }
        }
        private static readonly object lockConfigureLog4netProperties = new ();
        private static void ConfigureLog4netProperties(string logDirectory, string logFile, ContextPropertiesBase properties, bool overwriteIfExists)
        {
            properties["threadid"] = Thread.CurrentThread.ManagedThreadId;
            if (overwriteIfExists || properties["logdirectory"] == null || properties["logfile"] == null)
            {
                properties["logdirectory"] = logDirectory?.Replace("\\", "/") ?? "";
                properties["logfile"] = logFile;
            }
        }


        /// <summary>
        /// make a random coin flip, and returns:
        ///     true if head
        ///     false if tail
        /// </summary>
        /// <returns></returns>
        public static bool RandomCoinFlip()
        {
            return new Random(RandomSeed()).NextDouble() > 0.5;
        }

        public static int RandomSeed()
        {
            var randomSeed = Guid.NewGuid().GetHashCode();
            return randomSeed;
        }


        public static string GetEncoding(string filename)
        {
            using (FileStream fs = File.OpenRead(filename))
            {
                Ude.CharsetDetector cdet = new ();
                cdet.Feed(fs);
                cdet.DataEnd();
                return cdet.Charset;
            }
        }

        /// <summary>
        /// process the log of a model to look for values after some specific token
        /// the last value found for a token is always the one to use
        /// </summary>
        /// <param name="lines"></param>
        /// <param name="indexValueAfterToken"></param>
        /// <param name="tokenAndMandatoryItemAfterToken"></param>
        /// <returns></returns>
        public static double[] ExtractValuesFromOutputLog(IEnumerable<string> lines, int indexValueAfterToken, params string[] tokenAndMandatoryItemAfterToken)
        {
            Debug.Assert(tokenAndMandatoryItemAfterToken.Length%2 == 0);
            var token = new string[tokenAndMandatoryItemAfterToken.Length / 2];
            var mandatoryItemAfterToken = new string[token.Length];
            for (int i = 0; i < tokenAndMandatoryItemAfterToken.Length; i += 2)
            {
                token[i / 2] = tokenAndMandatoryItemAfterToken[i];
                mandatoryItemAfterToken[i / 2] = tokenAndMandatoryItemAfterToken[i + 1];
            }

            var results = Enumerable.Repeat(double.NaN, token.Length).ToArray();
            foreach(var line in lines.Reverse())
            {
                if (string.IsNullOrEmpty(line))
                {
                    continue;
                }
                if (results.All(val => !double.IsNaN(val)))
                {
                    return results; //we already have filled all values, no need to look in other lines
                }
                for (var j = 0; j < token.Length; j++)
                {
                    if (!double.IsNaN(results[j]))
                    {
                        continue; //we have already filled the value for token 'token[j]'
                    }
                    int idx = line.IndexOf(token[j], StringComparison.Ordinal);
                    if (idx < 0)
                    {
                        continue;
                    }
                    var splitted = line.Substring(idx + token[j].Length).Trim().Split();
                    if (   indexValueAfterToken< splitted.Length
                           && (mandatoryItemAfterToken[j] == null || mandatoryItemAfterToken[j] == splitted[0])
                           && double.TryParse(splitted[indexValueAfterToken], out var d))
                    {
                        results[j] = d;
                    }
                }
            }
            return results;
        }

        public static String UpdateFilePathWithPrefixSuffix(string filePath, string prefix, string suffix)
        {
            string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(filePath);
            string extension = Path.GetExtension(filePath);
            string path = GetDirectoryName(filePath);
            return ConcatenatePathWithFileName(path, prefix + fileNameWithoutExtension + suffix + extension);
        }
        public static bool FileExist(string fileName)
        {
            return !string.IsNullOrEmpty(fileName) && File.Exists(fileName);
        }
        public static int Max(int a, int b, int c, int d)
        {
            return Math.Max(Math.Max(a, b), Math.Max(c, d));
        }
        /// <summary>
        /// return the modulo of 'x' always in positive range [0, modulo-1]
        /// (even if x is negative)
        /// </summary>
        /// <param name="x">a number that can be negative</param>
        /// <param name="modulo"></param>
        /// <returns></returns>
        public static int AlwaysPositiveModulo(int x, int modulo)
        {
            int r = x % modulo;
            return r < 0 ? r + modulo : r;
        }

        public static string ToPython(bool b)
        {
            return b?"True":"False";
        }
    }
}
