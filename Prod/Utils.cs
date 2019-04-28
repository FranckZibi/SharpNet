using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.Data;

namespace SharpNet
{
    public class Logger
    {
        #region fields
        private readonly string _logFileName;
        private readonly bool _logInConsole;
        public static readonly Logger ConsoleLogger = new Logger("", true);
        public static readonly Logger NullLogger = new Logger("", false);
        #endregion

        public Logger(string logFileName, bool logInConsole)
        {
            _logFileName = logFileName ?? "";
            _logInConsole = logInConsole;
        }
        public void Info(string msg)
        {
            if (_logInConsole)
            {
                Console.WriteLine(msg);
            }
            LogInFile(msg);
        }
        public void Debug(string msg)
        {
            LogInFile(msg);
        }
        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(_logFileName), _logFileName)
                .Add(nameof(_logInConsole), _logInConsole)
                .ToString();
        }
        public static Logger ValueOf(IDictionary<string, object> serialized)
        {
            var logFileName = (string)serialized[nameof(_logFileName)];
            var logInConsole = (bool)serialized[nameof(_logInConsole)];
            return new Logger(logFileName, logInConsole);
        }
        private static string GetLinePrefix()
        {
            return DateTime.Now.ToString("HH:mm:ss.ff") + " ";
        }
        private void LogInFile(string msg)
        {
            if (string.IsNullOrEmpty(_logFileName))
            {
                return;
            }
            lock (_logFileName)
            {
                Utils.AddLineToFile(_logFileName, GetLinePrefix() + msg);
            }
        }
    }

    public static class Utils
    {
        public static void AddLineToFile(string filePath, string s)
        {
            var f = new FileInfo(filePath);
            if (f.Directory != null && !f.Directory.Exists)
            {
                f.Directory.Create();
            }
            using (var writer = new StreamWriter(f.FullName, true, Encoding.ASCII))
            {
                writer.WriteLine(s);
            }
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
        public static string ShapeToStringWithBacthSize(int[] shape)
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
            if (bytes > 3000000)
            {
                return (bytes / 1000000) + "MB";
            }
            if (bytes > 3000)
            {
                return (bytes / 1000) + "KB";
            }
            return bytes + "B";
        }
        public static double Interpolate(double x1, double y1, double x2, double y2, double xToInterpolate)
        {
            double dEpoch = (xToInterpolate - x1) / (x2 - x1);
            double deltaLearningRate = (y2 - y1);
            return y1 + dEpoch * deltaLearningRate;
        }
        public static void Randomize(double[] toRandomize, Random rand, double minValue, double maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = minValue + rand.NextDouble() * (maxValue - minValue);
            }
        }
        public static void Randomize(float[] toRandomize, Random rand, double minValue, double maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (float)(minValue + rand.NextDouble() * (maxValue - minValue));
            }
        }
        public static void RandomizeNormalDistribution(double[] toRandomize, Random rand, double mean, double stdDev)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = NextDoubleNormalDistribution(rand, mean, stdDev);
            }
        }
        public static void RandomizeNormalDistribution(float[] toRandomize, Random rand, double mean, double stdDev)
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

        //TODO :add tests
        public static bool TryGet<T>(this IDictionary<string, object> serialized, string key, out T value)
        {
            if (serialized.TryGetValue(key, out var resAsObject))
            {
                value = (T)resAsObject;
                return true;
            }
            value = default(T);
            return false;
        }
        public static T TryGet<T>(this IDictionary<string, object> serialized, string key)
        {
            if (serialized.TryGetValue(key, out var resAsObject))
            {
                return (T)resAsObject;
            }
            return default(T);
        }

        private static double NextDoubleNormalDistribution(Random rand, double mean, double stdDev)
        {
            //uniform(0,1) random double
            var u1 = rand.NextDouble();
            //uniform(0,1) random double
            var u2 = rand.NextDouble();
            //random normal(0,1)
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *Math.Sin(2.0 * Math.PI * u2);
            //random normal(mean,stdDev^2)
            return mean + stdDev * randStdNormal;
        }

        public static Logger Logger(string networkName)
        {
            var logFileName = ConcatenatePathWithFileName(@"c:\temp\ML\",
                networkName + "_" + Process.GetCurrentProcess().Id + "_" +
                System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            return new Logger(logFileName, true);
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

        public static bool Equals<T>(T a, T b, string id, ref string errors)
        {
            if (!Equals(a, b))
            {
                errors += id + ": " + a + " != " + b + Environment.NewLine;
                return false;
            }
            return true;
        }
        public static bool EqualsList(IList<int> a, IList<int> b, string id, ref string errors)
        {
            if (!a.SequenceEqual(b))
            {
                errors += id + ": " + string.Join(",", a) + " != " + string.Join(",", b) + Environment.NewLine;
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
        public static bool EqualsList(IList<Tensor> a, IList<Tensor> b, double epsilon, string id, ref string errors)
        {
            var tensorNames = new HashSet<string>(a.Where(x => x != null).Select(x => x.Description).Union(b.Where(x => x != null).Select(x => x.Description)));
            var allAreOk = true;
            foreach (var name in tensorNames.ToList())
            {
                allAreOk &= EqualsSingleTensor(a, b, name, epsilon, id, ref errors);
            }
            return allAreOk;
        }

        public static bool EqualsSingleTensor(IEnumerable<Tensor> a, IEnumerable<Tensor> b, string tensorName, double epsilon, string id, ref string errors)
        {
            var aFirst = a.FirstOrDefault(x => x.Description == tensorName);
            var bFirst = b.FirstOrDefault(x => x.Description == tensorName);
            return aFirst.Equals(bFirst, epsilon, id, ref errors);
        }
    }
}
