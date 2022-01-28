using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.MathTools;
using SharpNet.Networks;

namespace SharpNet.Datasets.CFM60
{
    public static class CFM60Utils
    {
        #region privater fields
        public static readonly Dictionary<int, CFM60Entry> id_to_entries = Load_Summary_File();
        #endregion

        public static float LinearRegressionEstimate(int id) => id_to_entries[id].LS;
        public static float LinearRegressionAdjustedByMeanEstimate(int id) => id_to_entries[id].NLV;

        private static Dictionary<int, double> ToPredictions(IEnumerable<CFM60Entry> entries, Func<CFM60Entry, double> ExtractPrediction = null)
        {
            var predictions = new Dictionary<int, double>();
            foreach (var e in entries)
            {
                predictions[e.ID] = (ExtractPrediction==null)?e.Y: ExtractPrediction(e);
            }
            return predictions;
        }

        public static double ComputeMse(IDictionary<int, double> predictions)
        {
            double sumSquareErrors = 0;
            foreach (var p in predictions)
            {
                var id = p.Key;
                var prediction = p.Value;
                if (!id_to_entries.ContainsKey(id))
                {
                    return double.NaN;
                }
                var Y = id_to_entries[id].Y;
                if (float.IsNaN(Y))
                {
                    return double.NaN;
                }
                var error = prediction - Y;
                sumSquareErrors += error * error;
            }

            var mse = sumSquareErrors / predictions.Count;
            return mse;
        }
        private static Dictionary<int, CFM60Entry> Load_Summary_File()
        {
            var res = new Dictionary<int, CFM60Entry>();
            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "CFM60_summary.csv");
            foreach (var l in File.ReadAllLines(path).Skip(1))
            {
                var entry = new CFM60Entry();
                var splitted = l.Split(',', ';');
                entry.ID = int.Parse(splitted[0]);
                entry.pid = int.Parse(splitted[1]);
                entry.day = int.Parse(splitted[2]);
                entry.Y = entry.LS = entry.NLV = float.NaN;
                if (float.TryParse(splitted[3], out var tmp))
                {
                    entry.Y = tmp;
                }
                if (splitted.Length >= 5)
                {
                    entry.LS = float.Parse(splitted[4]); //we store in the LS field the linear regression of Y
                }
                if (splitted.Length >= 6)
                {
                    entry.NLV = float.Parse(splitted[5]); //we store in the NLV field the adjusted linear regression of Y
                }
                res[entry.ID] = entry;
            }
            return res;
        }

        public static IDictionary<int, LinearRegression> ComputePidToLinearRegressionBetweenDayAndY(IEnumerable<CFM60Entry> entries)
        {
            var pidToLinearRegression = new Dictionary<int, LinearRegression>();
            foreach (var e in entries)
            {
                if (!pidToLinearRegression.ContainsKey(e.pid))
                {
                    pidToLinearRegression[e.pid] = new LinearRegression();
                }
                if (float.IsNaN(e.Y))
                {
                    continue;
                }
                pidToLinearRegression[e.pid].Add(e.day, e.Y);
            }
            return pidToLinearRegression;
        }

        public static int DayThreshold(IList<CFM60Entry> entries, double percentageInTrainingSet)
        {
            var sortedDays = entries.Select(e => e.day).OrderBy(x => x).ToArray();
            var countInTraining = (int)(percentageInTrainingSet * entries.Count);
            var dayThreshold = sortedDays[countInTraining];
            return dayThreshold;
        }
        
        public static IDictionary<int, double> LoadPredictions(string datasetPath)
        {
            var res = new Dictionary<int, double>();
            foreach (var l in File.ReadAllLines(datasetPath).Skip(1))
            {
                var splitted = l.Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries);
                var ID = int.Parse(splitted[0]);
                var Y = double.Parse(splitted[1]);
                res[ID] = Y;
            }
            return res;
        }


        public static void SavePredictions(IDictionary<int, double> CFM60EntryIDToPrediction, string filePath, double multiplierCorrection = 1.0, double addCorrectionStart = 0.0, double addCorrectionEnd = 0.0)
        {
            var sb = new StringBuilder();
            sb.Append("ID,target");
            foreach (var (id, originalPrediction) in CFM60EntryIDToPrediction.OrderBy(x => x.Key))
            {
                if (CFM60Entry.IsInterpolatedId(id))
                {
                    continue;
                }
                var day = id_to_entries[id].day;
                var fraction = (day - 805.0) / (1151 - 805.0);
                var toAdd = addCorrectionStart + (addCorrectionEnd - addCorrectionStart) * fraction;
                var prediction = multiplierCorrection * originalPrediction + toAdd;
                sb.Append(Environment.NewLine + id + "," + prediction.ToString(CultureInfo.InvariantCulture));
            }

            File.WriteAllText(filePath, sb.ToString());
        }

        public static float NormalizeBetween_0_and_1(float initialValue, float knownMinValue, float knownMaxValue)
        {
            if (knownMinValue >= knownMaxValue)
            {
                return 0; //constant
            }

            return (initialValue - knownMinValue) / (knownMaxValue - knownMinValue);
        }
    }
}
