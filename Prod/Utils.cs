using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using SharpNet.Data;

namespace SharpNet
{
    public class Logger
    {
        private readonly string _logFileName;
        private readonly bool _logInConsole;
        public Logger(string logFileName, bool logInConsole)
        {
            _logFileName = logFileName??"";
            _logInConsole = logInConsole;
        }

        public static readonly Logger ConsoleLogger = new Logger("", true);
        public static readonly Logger NullLogger = new Logger("", false);

        public void Info(string msg)
        {
            if (_logInConsole)
            {
                Console.WriteLine(msg);
            }
            if (!string.IsNullOrEmpty(_logFileName))
            {
                lock (_logFileName)
                {
                    Utils.AddLineToFile(_logFileName, GetLinePrefix() + msg);
                }
            }
        }
        public void Debug(string msg)
        {
            if (!string.IsNullOrEmpty(_logFileName))
            {
                lock (_logFileName)
                {
                    Utils.AddLineToFile(_logFileName, GetLinePrefix() + msg);
                }
            }
        }
        public void Error(string msg) { Info(msg); }
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
    }


    public static class Utils
    {
        [DllImport("Kernel32.dll", EntryPoint = "RtlZeroMemory", SetLastError = false)]
        public static extern void ZeroMemory(IntPtr dest, size_t sizeInBytes);

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

     
        //TODO add tests
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
            return "(batchSize, " + string.Join(", ", shape.Skip(1)) + ")";
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
        private static double NextDoubleNormalDistribution(Random rand, double mean, double stdDev)
        {
            double u1 = rand.NextDouble(); //these are uniform(0,1) random doubles
            double u2 = rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                   Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
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

        public static String UpdateFilePathChangingExtension(string filePath, string prefix, string suffix, string newExtension)
        {
            string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(filePath);
            if (!newExtension.StartsWith("."))
            {
                newExtension = "." + newExtension;
            }
            string path = GetDirectoryName(filePath);
            return ConcatenatePathWithFileName(path, prefix + fileNameWithoutExtension + suffix + newExtension);
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
        public static string ConcatenatePathWithFileName(string path, params string[] subPaths)
        {
            string result = path;
            foreach (var t in subPaths)
            {
                result = Path.Combine(result, t);
            }
            return result;
        }
    }
}
