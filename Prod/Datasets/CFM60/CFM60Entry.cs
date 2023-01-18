using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using ProtoBuf;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.CFM60
{
    [ProtoContract]
    public class CFM60Entry : TimeSeriesSinglePoint
    {
        public const int POINTS_BY_DAY = 61;
        public const int DISTINCT_PID_COUNT = 900;  //total number of distinct companies (pid)


        public static bool IsInterpolatedId(int id) {return id < 0;}


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
            LS = (float)double.Parse(splitted[index++], CultureInfo.InvariantCulture);
            NLV = (float)double.Parse(splitted[index], CultureInfo.InvariantCulture);
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
            foreach (var (id, y) in CFM60DatasetSample.LoadPredictions(yFileIfAny))
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

        public string UniqueId => ID.ToString();
        public string TimeSeriesFamily => pid.ToString();
        public float TimeSeriesTimeStamp => day;
        public float ExpectedTarget => Y;
    }
}