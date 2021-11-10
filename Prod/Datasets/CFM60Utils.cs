using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.MathTools;
using SharpNet.Networks;

namespace SharpNet.Datasets
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

        public static void Create_Summary_File(IList<CFM60Entry> entries)
        {
            var pid_to_linear_regression = ComputePidToLinearRegressionBetweenDayAndY(entries);

            // predictions based on linear regression only
            var preds_lr = new Dictionary<int, double>();
            foreach (var e in entries)
            {
                preds_lr[e.ID] = pid_to_linear_regression[e.pid].Estimation(e.day);
            }

            // predictions based on linear regression adjusted by taking into account the true mean by interval
            var preds_lr_mean_adjusted = AdjustPredictionFromTargetMean(preds_lr);

            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "CFM60_summary.csv");
            var sb = new StringBuilder();
            sb.Append("ID;pid;day;Y;Y_lr;Y_lr_fixed" + Environment.NewLine);
            foreach (var e in entries.OrderBy(e => e.ID))
            {
                sb.Append(e.ID 
                          + ";" + e.pid
                          + ";" + e.day
                          + ";" + (float.IsNaN(e.Y) ? "" : e.Y.ToString(CultureInfo.InvariantCulture))
                          + ";" + ((float)preds_lr[e.ID]).ToString(CultureInfo.InvariantCulture)
                          + ";" + ((float)preds_lr_mean_adjusted[e.ID]).ToString(CultureInfo.InvariantCulture)
                          + Environment.NewLine);
            }
            File.WriteAllText(path, sb.ToString());
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

        public static List<CFM60Entry> GetTestPredictions()
        {
            return id_to_entries.Values.Where(e => float.IsNaN(e.Y)).OrderBy(e => e.ID).ToList();
        }

        public static List<CFM60Entry> GetTrainPredictions()
        {
            return id_to_entries.Values.Where(e => !float.IsNaN(e.Y)).OrderBy(e => e.ID).ToList();
        }

        public static Tuple<List<CFM60Entry>, List<CFM60Entry>> GetTrainAndValidationEntries(double percentageInTrainingSet)
        {
            var allTrain = GetTrainPredictions();
            var dayThreshold = DayThreshold(allTrain, percentageInTrainingSet);
            var usedForTraining = allTrain.Where(e => e.day <= dayThreshold).ToList();
            var usedForValidation = allTrain.Where(e => e.day > dayThreshold).ToList();
            return Tuple.Create(usedForTraining, usedForValidation);
        }

        public static void CreateTestPredictionFileForAverage()
        {
            var id2Day = new Dictionary<int, int>();
            foreach (var e in GetTestPredictions())
            {
                id2Day[e.ID] = e.day;
            }
            foreach (var i in new[]
            {
                Tuple.Create(805, 810),
            })
            {
                var predictions = new Dictionary<int, double>();
                var countInInterval = 0;
                foreach (var e in id2Day)
                {
                    bool isInInterval = e.Value >= i.Item1 && e.Value <= i.Item2;
                    predictions[e.Key] = isInInterval ? 1.0 : 0.0;
                    if (isInInterval)
                    {
                        ++countInInterval;
                    }
                }
                var path = Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "dump", "preds_" + i.Item1 + "_" + i.Item2 + "_count_" + countInInterval + ".csv");
                SavePredictions(predictions, path);
            }
        }

        private static IDictionary<Tuple<int,int>, double> Load_Intervals_To_Mean()
        {
            var res = new Dictionary<Tuple<int, int>, double>();
            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "mean_by_intervals.csv");
            foreach (var l in File.ReadAllLines(path).Skip(1))
            {
                var splitted = l.Split(',', ';').ToArray();
                var start = int.Parse(splitted[0]);
                var end = int.Parse(splitted[1]);
                Debug.Assert(start <= end);
                var mean = double.Parse(splitted[2]);
                res[Tuple.Create(start, end)] = mean;
            }
            return res;
        }

        public static void Create_Intervals_To_Mean_For_Training(int interval_length)
        {
            var trainPredictions = GetTrainPredictions();
            int min_day = trainPredictions.Select(e => e.day).Min();
            int max_day = trainPredictions.Select(e => e.day).Max();
            var specificDays = CFM60DataSet.EndOfTrimester.Union(CFM60DataSet.EndOfYear).Union(CFM60DataSet.Christmas).ToList();
            specificDays.Sort();
            int nextDay = min_day;
            var intervals =new List<Tuple<int, int>>();
            while (nextDay <= max_day)
            {
                if (specificDays.Count != 0 && nextDay >= specificDays[0])
                {
                    intervals.Add(Tuple.Create(specificDays[0], specificDays[0]));
                    specificDays.RemoveAt(0);
                }
                else
                {
                    int endDay = nextDay + interval_length - 1;
                    endDay = Math.Min(endDay, max_day);
                    endDay = Math.Min(endDay, specificDays[0]-1);
                    intervals.Add(Tuple.Create(nextDay, endDay));
                }
                nextDay = intervals.Last().Item2+1;
            }
            var truePredictions = ToPredictions(trainPredictions);
            var days2Interval = Days2Interval(truePredictions, intervals);
            var interval2Acc = Interval2Acc(truePredictions, days2Interval);
            var sb = new StringBuilder();
            sb.Append("day_start;day_end;mean" + Environment.NewLine);
            foreach (var i in intervals)
            {
                sb.Append(i.Item1 + ";" + i.Item2 + ";" + interval2Acc[i].Average.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
            }
            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "mean_by_intervals_train_"+interval_length+".csv");
            File.WriteAllText(path, sb.ToString());
        }

        private static IDictionary<int, Tuple<int, int>> Days2Interval(IDictionary<int, double> IDToPrediction, List<Tuple<int, int>> intervals)
        {
            var res = new Dictionary<int, Tuple<int, int>>();
            foreach (var p in IDToPrediction)
            {
                var ID = p.Key;
                var day = id_to_entries[ID].day;
                if (!res.ContainsKey(day))
                {
                    res[day] = Day2Interval(day, intervals);
                }
            }
            return res;
        }
        private static Tuple<int, int> Day2Interval(int day, IEnumerable<Tuple<int, int>> intervals)
        {
            foreach (var i in intervals)
            {
                if (day >= i.Item1 && day <= i.Item2)
                {
                    return i;
                }
            }
            throw new Exception("invalid day " + day);
        }

        private static IDictionary<Tuple<int, int>, DoubleAccumulator> Interval2Acc(IDictionary<int, double> IDToPrediction, IDictionary<int, Tuple<int, int>> day2Interval)
        {
            var res = new Dictionary<Tuple<int, int>, DoubleAccumulator>();
            foreach (var p in IDToPrediction)
            {
                var ID = p.Key;
                var prediction = p.Value;
                var day = id_to_entries[ID].day;
                var interval = day2Interval[day];
                if (!res.ContainsKey(interval))
                {
                    res[interval] = new DoubleAccumulator();
                }
                res[interval].Add(prediction, 1);
            }
            return res;
        }
        
        public static IDictionary<int, double> AdjustPredictionFromTargetMean(IDictionary<int, double> IDToPrediction)
        {
            var intervals_To_Mean = Load_Intervals_To_Mean();
            var intervals = intervals_To_Mean.Keys.ToList();
            var day2Interval = Days2Interval(IDToPrediction, intervals);
            var interval2Acc = Interval2Acc(IDToPrediction, day2Interval);
            foreach (var i in interval2Acc.Keys.ToList())
            {
                Console.WriteLine("In interval "+i+": observed avg:" + interval2Acc[i].Average+ " ; target avg:" + intervals_To_Mean[i]);
            }

            var fixedPredictions = new Dictionary<int, double>();
            foreach (var originalPredictions in IDToPrediction)
            {
                var ID = originalPredictions.Key;
                var originalPrediction = originalPredictions.Value;
                var day = id_to_entries[ID].day;
                var interval = day2Interval[day];
                var originalAverage = interval2Acc[interval].Average;
                var targetAverage = intervals_To_Mean[interval];
                var fixedPrediction = originalPrediction + (targetAverage - originalAverage);
                fixedPredictions[ID] = fixedPrediction;
            }
            return fixedPredictions;
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

        public static IDictionary<int, double>  AdjustPredictionsWithLinearDeformation(IDictionary<int, double> predictions, double multiplierCorrection = 1.0, double addCorrectionStart = 0.0, double addCorrectionEnd = 0.0)
        {
            var res = new Dictionary<int, double>();
            var min_day = predictions.Keys.Select(id => id_to_entries[id].day).Min();
            var max_day = predictions.Keys.Select(id => id_to_entries[id].day).Max();
            foreach (var (id, originalPrediction) in predictions.OrderBy(x => x.Key))
            {
                res[id] = originalPrediction;
                var day = id_to_entries[id].day;
                var fraction = ((double)day - min_day) / ((double)max_day - min_day);
                var toAdd = addCorrectionStart + (addCorrectionEnd - addCorrectionStart) * fraction;
                res[id] = multiplierCorrection * originalPrediction + toAdd;
            }
            return res;
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
