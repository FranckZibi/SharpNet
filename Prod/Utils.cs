using System;
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
    public static partial class Utils
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(Utils));


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
