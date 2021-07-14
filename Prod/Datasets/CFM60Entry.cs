using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Linq;
using ProtoBuf;
using SharpNet.MathTools;

namespace SharpNet.Datasets
{
    [ProtoContract]
    public class CFM60Entry
    {
        public const int POINTS_BY_DAY = 61;
        public const int DISTINCT_PID_COUNT = 900;  //total number of distinct companies (pid)


        private float? _volatility_ret_vol = null;
        private float? _volatility_abs_ret = null;
        private float? _mean_abs_ret = null;
        private float? _ret_vol_CoefficientOfVariation = null;

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
            ret_vol = new float[POINTS_BY_DAY];
            for (int i = 0; i < POINTS_BY_DAY; ++i)
            {
                if (double.TryParse(splitted[index++], out var tmp))
                {
                    ret_vol[i] = (float)tmp;
                }
            }
            LS = (float)double.Parse(splitted[index++]);
            NLV = (float)double.Parse(splitted[index]);
        }

        [ProtoMember(1)]
        public int ID { get;  }
        [ProtoMember(2)]
        public int pid { get; }
        [ProtoMember(3)]
        public int day { get; }
        [ProtoMember(4)]
        public float[] abs_ret { get; }
        [ProtoMember(5)]
        public float[] ret_vol{ get; }
        [ProtoMember(6)]
        public float LS { get; }
        [ProtoMember(7)]
        public float NLV { get; }
        [ProtoMember(8)] 
        public float Y { get; private set; } = float.NaN;

        public static CFM60Entry[] Load(string xFile, string yFileIfAny, Action<string> log)
        {
            Debug.Assert(xFile != null);

            var protoBufFile = xFile + ".proto";
            if (File.Exists(protoBufFile))
            {
                log("Loading ProtoBuf file " + protoBufFile + "...");
                using var fsLoad = new FileStream(protoBufFile, FileMode.Open, FileAccess.Read);
                var result = Serializer.Deserialize<CFM60Entry[]>(fsLoad);
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
            foreach (var (id, y) in CFM60DataSet.LoadPredictionFile(yFileIfAny))
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
        public float Get_volatility_ret_vol()
        {
            if (!_volatility_ret_vol.HasValue)
            {
                var acc = new DoubleAccumulator();
                acc.Add(ret_vol);
                _volatility_ret_vol = (float)acc.Volatility;
            }
            return _volatility_ret_vol.Value;
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

        public float Get_ret_vol_CoefficientOfVariation()
        {
            if (!_ret_vol_CoefficientOfVariation.HasValue)
            {
                var acc = new DoubleAccumulator();
                acc.Add(ret_vol);
                _ret_vol_CoefficientOfVariation = (float)acc.CoefficientOfVariation;
            }
            return _ret_vol_CoefficientOfVariation.Value;
        }

    }
}