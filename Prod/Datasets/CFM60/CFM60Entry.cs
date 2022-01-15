using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using ProtoBuf;
using SharpNet.MathTools;
using SharpNet.Networks;

namespace SharpNet.Datasets.CFM60
{
    [ProtoContract]
    public class CFM60Entry
    {
        public const int POINTS_BY_DAY = 61;
        public const int DISTINCT_PID_COUNT = 900;  //total number of distinct companies (pid)


        private static float Interpolate(CFM60Entry before, CFM60Entry after, int dayToInterpolate, Func<CFM60Entry, float> extractValue)
        {
            if (after == null)
            {
                return extractValue(before);
            }
            if (before == null)
            {
                return extractValue(after);
            }
            Debug.Assert(before.pid == after.pid);
            Debug.Assert(before.day < after.day);
            Debug.Assert(dayToInterpolate>before.day);
            Debug.Assert(dayToInterpolate<after.day);
            var beforeValue = extractValue(before);
            var afterValue = extractValue(after);
            if (float.IsNaN(afterValue))
            {
                return beforeValue;
            }
            if (float.IsNaN(beforeValue))
            {
                return afterValue;
            }
            return beforeValue + ((dayToInterpolate-before.day) / ((float)(after.day - before.day))) * (afterValue - beforeValue);
        }
        private static float[] Interpolate(CFM60Entry before, CFM60Entry after, int dayToInterpolate, Func<CFM60Entry, float[]> extractValue)
        {
            if (after == null)
            {
                return (float[])extractValue(before).Clone();
            }
            if (before == null)
            {
                return (float[])extractValue(after).Clone();
            }
            var result = new float[extractValue(before).Length];
            for (int i = 0; i < result.Length; ++i)
            {
                var iCopy = i;
                result[i] = Interpolate(before, after, dayToInterpolate, e => extractValue(e)[iCopy]);
            }
            return result;
        }
        public static bool IsInterpolatedId(int id) {return id < 0;}



        public static CFM60Entry Interpolate(CFM60Entry before, CFM60Entry after, int dayToInterpolate)
        {
            return new CFM60Entry
                        {
                            ID = -(Math.Abs(before.ID)+1),
                            pid = before.pid,
                            day = dayToInterpolate,
                            abs_ret = Interpolate(before, after, dayToInterpolate, e => e.abs_ret),
                            rel_vol = Interpolate(before, after, dayToInterpolate, e => e.rel_vol),
                            LS = Interpolate(before, after, dayToInterpolate, e => e.LS),
                            NLV = Interpolate(before, after, dayToInterpolate, e => e.NLV),
                            Y = Interpolate(before, after, dayToInterpolate, e => e.Y)
                        };
        }

        private float? _volatility_rel_vol = null;
        private float? _volatility_abs_ret = null;
        private float? _mean_abs_ret = null;
        private float? _rel_vol_CoefficientOfVariation = null;

        /// <summary>
        /// parameter less constructor needed for ProtoBuf serialization 
        /// </summary>
        // ReSharper disable once UnusedMember.Global
        public CFM60Entry() { }

        private CFM60Entry(string line)
        {
            var splitted = line.Split(',');
            int index = 0;
            ID = int.Parse(splitted[index++]);
            pid = int.Parse(splitted[index++]);
            day = int.Parse(splitted[index++]);
            abs_ret = new float[POINTS_BY_DAY];
            for (int i = 0; i < POINTS_BY_DAY; ++i)
            {
                if (double.TryParse(splitted[index++], out var tmp))
                {
                    abs_ret[i] = (float)tmp;
                }
            }
            rel_vol = new float[POINTS_BY_DAY];
            for (int i = 0; i < POINTS_BY_DAY; ++i)
            {
                if (double.TryParse(splitted[index++], out var tmp))
                {
                    rel_vol[i] = (float)tmp;
                }
            }
            LS = (float)double.Parse(splitted[index++]);
            NLV = (float)double.Parse(splitted[index]);
        }

        [ProtoMember(1)]
        public int ID { get; set; }
        [ProtoMember(2)]
        public int pid { get; set; }
        [ProtoMember(3)]
        public int day { get; set; }
        [ProtoMember(4)]
        public float[] abs_ret { get; set; }
        [ProtoMember(5)]
        public float[] rel_vol{ get; set; }
        [ProtoMember(6)]
        public float LS { get; set; }
        [ProtoMember(7)]
        public float NLV { get; set; }
        [ProtoMember(8)] 
        public float Y { get; set; } = float.NaN;

        public static void ReplaceYByError(CFM60Entry[] entries, Action<string> log, string[] predictionFilesIfComputeErrors)
        {
            Debug.Assert(predictionFilesIfComputeErrors!=null);
            var idToEntry = new Dictionary<int, CFM60Entry>();
            foreach (var e in entries)
            {
                idToEntry[e.ID] = e;
            }
            log("trying to reduce errors");
            foreach (var f in predictionFilesIfComputeErrors)
            {
                int count = 0;
                int missed = 0;
                double totalError = 0;
                foreach (var (id, prediction) in CFM60Utils.LoadPredictions(f))
                {
                    ++count;
                    if (!idToEntry.ContainsKey(id))
                    {
                        ++missed;
                        continue;
                    }
                    var entry = idToEntry[id];
                    var error = entry.Y - prediction;
                    entry.Y = (float)error;
                    totalError += Math.Abs(error);
                }
                if (missed != 0 && missed != count)
                {
                    throw new Exception("invalid file " + f);
                }
                log("Mae = "+ (totalError / Math.Max(count, 1)) + " for " + f);
            }
        }

        public static void ReplaceYByLinearRegressionEstimate(CFM60Entry[] entries)
        {
            foreach (var e in entries)
            {
                e.Y = e.Y - CFM60Utils.LinearRegressionEstimate(e.ID);
            }
        }
        public static void ReplaceYByLinearRegressionAdjustedByMeanEstimate(CFM60Entry[] entries)
        {
            foreach (var e in entries)
            {
                e.Y = e.Y - CFM60Utils.LinearRegressionAdjustedByMeanEstimate(e.ID);
            }
        }

        public static CFM60Entry[] Load(string xFile, string yFileIfAny, Action<string> log, CFM60NetworkBuilder.ValueToPredictEnum valueToPredict, string[] predictionFilesIfComputeErrors)
        {
            Debug.Assert(xFile != null);

            var protoBufFile = xFile + ".proto";
            if (File.Exists(protoBufFile))
            {
                log("Loading ProtoBuf file " + protoBufFile + "...");
                using var fsLoad = new FileStream(protoBufFile, FileMode.Open, FileAccess.Read);
                var result = Serializer.Deserialize<CFM60Entry[]>(fsLoad);
                if (valueToPredict == CFM60NetworkBuilder.ValueToPredictEnum.ERROR)
                {
                    ReplaceYByError(result, log, predictionFilesIfComputeErrors);
                }
                else if (valueToPredict == CFM60NetworkBuilder.ValueToPredictEnum.Y_TRUE_MINUS_LR)
                {
                    log("Trying to predict error compared to linear regression estimate");
                    ReplaceYByLinearRegressionEstimate(result);
                }
                else if (valueToPredict == CFM60NetworkBuilder.ValueToPredictEnum.Y_TRUE_MINUS_ADJUSTED_LR)
                {
                    log("Trying to predict error compared to adjusted linear regression estimate");
                    ReplaceYByLinearRegressionAdjustedByMeanEstimate(result);
                }
                log("Binary file " + protoBufFile + " has been loaded");
                return result;
            }

            log("Loading content of file " + xFile + "...");
            var xFileLines = File.ReadAllLines(xFile);

            log("File " + xFile + " has been loaded: " + xFileLines.Length + " lines");

            log("Parsing lines of file " + xFile + "...");
            var entries = new CFM60Entry[xFileLines.Length - 1];
            var idToIndex = new ConcurrentDictionary<int, int>();
            void ProcessLine(int i)
            {
                Debug.Assert(i >= 1);
                entries[i - 1] = new CFM60Entry(xFileLines[i]);
                idToIndex[entries[i - 1].ID] = i - 1;
            }
            System.Threading.Tasks.Parallel.For(1, xFileLines.Length, ProcessLine);
            log("Lines of file " + xFile + " have been parsed");
            foreach (var (id, y) in CFM60Utils.LoadPredictions(yFileIfAny))
            {
                entries[idToIndex[id]].Y = (float)y;
            }

          

            log("Writing ProtoBuf file " + protoBufFile + "...");
            using var fs = new FileStream(protoBufFile, FileMode.Create);
            Serializer.Serialize(fs, entries);
            fs.Close();
            log("ProtoBuf file " + protoBufFile + " has been loaded");
            return entries;
        }
        public float Get_volatility_rel_vol()
        {
            if (!_volatility_rel_vol.HasValue)
            {
                var acc = new DoubleAccumulator();
                acc.Add(rel_vol);
                _volatility_rel_vol = (float)acc.Volatility;
            }
            return _volatility_rel_vol.Value;
        }

        public float Get_mean_abs_ret()
        {
            if (!_mean_abs_ret.HasValue)
            {
                _mean_abs_ret = abs_ret.Sum() / abs_ret.Length;
            }
            return _mean_abs_ret.Value;
        }
        public float Get_volatility_abs_ret()
        {
            if (!_volatility_abs_ret.HasValue)
            {
                var acc = new DoubleAccumulator();
                acc.Add(abs_ret);
                _volatility_abs_ret = (float)acc.Volatility;
            }
            return _volatility_abs_ret.Value;
        }

        public float Get_rel_vol_CoefficientOfVariation()
        {
            if (!_rel_vol_CoefficientOfVariation.HasValue)
            {
                var acc = new DoubleAccumulator();
                acc.Add(rel_vol);
                _rel_vol_CoefficientOfVariation = (float)acc.CoefficientOfVariation;
            }
            return _rel_vol_CoefficientOfVariation.Value;
        }
   
    }
}