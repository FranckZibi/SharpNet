using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace SharpNet
{
    public static partial class Utils
    {
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
    }
}
