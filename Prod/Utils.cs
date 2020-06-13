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
using SharpNet.Networks;
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
        public static double Interpolate(List<Tuple<double,double>> values, double x, bool constantByInterval = false)
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
        public static void Randomize(Span<float> toRandomize, Random rand, double minValue, double maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (float)(minValue + rand.NextDouble() * (maxValue - minValue));
            }
        }
        public static void Randomize(Span<byte> toRandomize, Random rand, byte minValue, byte maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (byte)(minValue + rand.Next(maxValue - minValue+1));
            }
        }
        public static void RandomizeNormalDistribution(Span<float> toRandomize, Random rand, double mean, double stdDev)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (float)NextDoubleNormalDistribution(rand, mean, stdDev);
            }
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
        public static string UpdateFilePathChangingExtension(string filePath, string prefix, string suffix, string newExtension)
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
                if ((alpha*tmp+(gamma*v)-1.3862944) >= Math.Log(u1*u1*u2))
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
        public static T2[] Select<T1,T2>(this ReadOnlySpan<T1> s, Func<T1, T2> select)
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
        public static bool All<T>(this ReadOnlySpan<T> s, Func<T, bool> isIncluded)
        {
            foreach (var t in s)
            {
                if (!isIncluded(t))
                {
                    return false;
                }
            }
            return true;
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

        public static void ConfigureGlobalLog4netProperties()
        {
            ConfigureLog4netProperties(NetworkConfig.DefaultLogDirectory, "SharpNet", GlobalContext.Properties);
        }


        public static void ConfigureThreadLog4netProperties(string logDirectory, string logFile)
        {
            ConfigureLog4netProperties(logDirectory, logFile, ThreadContext.Properties);
            XmlConfigurator.Configure(LogManager.GetRepository(Assembly.GetEntryAssembly()), new FileInfo("log4net.config"));
        }

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

    }
}
