using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Xml;
using log4net;
using log4net.Config;
using log4net.Util;
using SharpNet.Pictures;

namespace SharpNet
{
    public static class Utils
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(Utils));

        public static string ToValidFileName(string fileName)
        {
            var invalids = new HashSet<char>(Path.GetInvalidFileNameChars());
            return new string(fileName.Select(x => invalids.Contains(x) ? '_' : x).ToArray());
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


        /// <summary>
        /// Make input 'rectangularMatrix' an orthogonal matrix using Gram–Schmidt process
        /// See: https://en.wikipedia.org/wiki/QR_decomposition
        /// </summary>
        /// <param name="rectangularMatrix">A rectangular matrix with shape: (rows, cols)
        /// that we want to make orthogonal</param>
        /// <param name="rows">number of rows in matrix 'rectangularMatrix'</param>
        /// <param name="cols">number of columns in matrix 'rectangularMatrix'</param>
        public static void ToOrthogonalMatrix(Span<float> rectangularMatrix, int rows, int cols)
        {
            Debug.Assert(rectangularMatrix.Length == rows * cols);

            //We compute the U matrix as described in: https://en.wikipedia.org/wiki/QR_decomposition 
            var U = new Span<float>(new float[rectangularMatrix.Length]);
            rectangularMatrix.CopyTo(U);
            for (int row = 1; row < rows; ++row)
            {
                //we compute row 'row' of 'U' matrix
                var aRow = rectangularMatrix.Slice(cols * row, cols);
                var uRow = U.Slice(cols * row, cols);
                for (int subRow = 0; subRow < row; ++subRow)
                {
                    var uSubRow = U.Slice(cols * subRow, cols);
                    float multiplier = InnerProduct(uSubRow, aRow) / InnerProduct(uSubRow, uSubRow);
                    for (int col = 0; col < uSubRow.Length; ++col)
                    {
                        uRow[col] -= multiplier * uSubRow[col];
                    }
                }
            }

            //We compute the Q (= rectangularMatrix) matrix:
            //  it is an orthogonal matrix that we can compute from the U matrix
            //  (by normalizing each row of the U matrix)
            U.CopyTo(rectangularMatrix);
            for (int row = 0; row < rows; ++row)
            {
                var aRow = rectangularMatrix.Slice(cols * row, cols);
                float normalizer = (float)Math.Sqrt(InnerProduct(aRow, aRow));
                for (int col = 0; col < aRow.Length; ++col)
                {
                    aRow[col] /= normalizer;
                }
            }
        }

        private static float InnerProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            Debug.Assert(a.Length == b.Length);
            float result = 0;
            for (int i = 0; i < a.Length; ++i)
            {
                result += a[i] * b[i];
            }

            return result;
        }

        public static void Shuffle<T>(IList<T> list, Random rand)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rand.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
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

        public static string ConcatenatePathWithFileName(string path, params string[] subPaths)
        {
            string result = path;
            foreach (var t in subPaths)
            {
                result = Path.Combine(result, t);
            }

            return result;
        }

        public static long FileLength(string path)
        {
            return new FileInfo(path).Length;
        }

        /// <summary>
        /// read a part of a binary file, starting at position 'startIndex' in the file
        /// </summary>
        /// <param name="fileName">file to read</param>
        /// <param name="startIndex">the fistIndex to read in the file</param>
        /// <param name="byteCount">number of bytes to read</param>
        /// <returns>an array of 'byteCount' bytes</returns>
        public static byte[] ReadPartOfFile(string fileName, int startIndex, int byteCount)
        {
            // Open file with a BinaryReader
            using (var b = new BinaryReader(File.Open(fileName, FileMode.Open, FileAccess.Read, FileShare.Read)))
            {
                // Seek to our required position 'startIndex'
                b.BaseStream.Seek(startIndex, SeekOrigin.Begin);
                // Read the next 'byteCount' bytes.
                return b.ReadBytes(byteCount);
            }
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

        public static T TryGet<T>(this IDictionary<string, object> serialized, string key)
        {
            if (serialized.TryGetValue(key, out var resAsObject))
            {
                return (T)resAsObject;
            }

            return default;
        }

        public static bool Equals<T>(T a, T b, string id, ref string errors)
        {
            if (!Equals(a, b))
            {
                errors += id + ": " + a + " != " + b + Environment.NewLine;
                return false;
            }

            return true;
        }

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

        private static float Sum(this ReadOnlySpan<float> s)
        {
            var result = 0f;
            foreach (var t in s)
            {
                result += t;
            }

            return result;
        }

        public static float Average(this ReadOnlySpan<float> s)
        {
            if (s == null || s.Length == 0)
            {
                return 0;
            }

            return Sum(s) / s.Length;
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

        public static void ConfigureGlobalLog4netProperties(string logDirectory, string logFile)
        {
            lock (lockConfigureLog4netProperties)
            {
                ConfigureLog4netProperties(logDirectory, logFile, GlobalContext.Properties);
                XmlConfigurator.Configure(LogManager.GetRepository(Assembly.GetEntryAssembly()),
                    new FileInfo(@"log4net.config"));
            }
        }

        public static void ConfigureThreadLog4netProperties(string logDirectory, string logFile)
        {
            lock (lockConfigureLog4netProperties)
            {
                ConfigureLog4netProperties(logDirectory, logFile, ThreadContext.Properties);
                XmlConfigurator.Configure(LogManager.GetRepository(Assembly.GetEntryAssembly()), new FileInfo(@"log4net.config"));
            }
        }

        private static readonly object lockConfigureLog4netProperties = new object();

        private static void ConfigureLog4netProperties(string logDirectory, string logFile, ContextPropertiesBase properties)
        {
            properties["threadid"] = Thread.CurrentThread.ManagedThreadId;
            properties["logdirectory"] = logDirectory?.Replace("\\", "/") ?? "";
            properties["logfile"] = logFile;
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
            using var sha1 = new SHA1Managed();
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

        public static bool SameContent(float[] a, float[] b, double epsilon)
        {
            if (a.Length != b.Length)
            {
                return false;
            }

            for (int i = 0; i < a.Length; ++i)
            {
                if (double.IsNaN(a[i]) != double.IsNaN(b[i]))
                {
                    return false;
                }

                if (Math.Abs(a[i] - b[i]) > epsilon)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        ///  version contains(1000 major + 100 minor + build).
        /// For example, 7.6.5 would be represented by 7605
        /// </summary>
        /// <param name="version"></param>
        /// <returns></returns>
        public static Version NewVersion(int version)
        {
            var major = version / 1000;
            var minor = (version % 1000) / 100;
            var build = version % 100;
            return new Version(major, minor, build);
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

        public static int CoreCount
        {
            get
            {
                int coreCount = 0;
                foreach (var item in new System.Management.ManagementObjectSearcher("Select * from Win32_Processor").Get())
                {
                    coreCount += int.Parse(item["NumberOfCores"].ToString());
                }
                return coreCount;
            }
        }


        /// <summary>
        /// Compute the % time to invest on each use case, knowing the error associated with each use case
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
        ///  => a value close to 0 means we want to invest very little time on this usecase (because it doesn't seem use full)
        /// </returns>
        public static double[] TargetCpuInvestmentTime(List<Tuple<double, double, int>> errors)
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

            var valueWithIndex = new List<Tuple<Tuple<double, double, int>, int>>();
            for (int i = 0; i < errors.Count; ++i)
            {
                if (errors[i].Item3 <= 1)
                {
                    //if the error is computed on 0 or 1 experiment, we can not rely on the associated error,
                    //so the associated use case will have the mean cpu time of all use cases
                    continue;
                }
                valueWithIndex.Add(Tuple.Create(errors[i], i));
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
            const double MinPonderation = 0.05;
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
                    weights.Add(MinPonderation);
                    continue;
                }

                double percentageInCommon = (lowestErrorSupMargin - currentErrorInfMargin) /  (lowestErrorSupMargin - lowestErrorInfMargin);
                double weight = percentageInCommon / (2 - percentageInCommon);
                Debug.Assert(weight<=1.0001);
                Debug.Assert(weight>=0.0);
                weights.Add(Math.Max(MinPonderation, weight));
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
    }

}
