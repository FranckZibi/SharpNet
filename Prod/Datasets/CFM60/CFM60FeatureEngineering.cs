using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.MathTools;

namespace SharpNet.Datasets.CFM60
{
    public static class CFM60FeatureEngineering
    {
        const int pid_count = 900;
        const int day_count = 1152;

        public static void Process()
        {
            var trainPredictions = CFM60Utils.LoadPredictions(@"C:\Temp\CFM60_30_0_3099_0_3595_20211024_1207_4_train.csv");
            var validationPredictions = CFM60Utils.LoadPredictions(@"C:\Temp\CFM60_30_0_3099_0_3595_20211024_1207_4_validation.csv");
            //var testPredictions = LoadPredictions(@"C:\Temp\CFM60_30_0_3099_0_3595_20211024_1207_4_test.csv");
            var testPredictions = CFM60Utils.LoadPredictions(@"C:\Temp\EnsembleLearning_637621601154307303_2x_network_actual_0_4069_add_ln_0then_addd_ln_1_12_actual_0_4050.csv");

            Console.WriteLine("trainPredictions mse: " + CFM60Utils.ComputeMse(trainPredictions));
            Console.WriteLine("validationPredictions mse: " + CFM60Utils.ComputeMse(validationPredictions));
            Console.WriteLine("testPredictions mse: " + CFM60Utils.ComputeMse(testPredictions));

            var errorsLinearRegression = ComputeLinearRegression(trainPredictions);

            foreach (var multiplicativeCoeff in new[] { 1 })
            {
                //var fixedTrainPredictions = ComputeFixedPredictions(trainPredictions, id_to_info, errorsLinearRegression, multiplicativeCoeff);
                //Console.WriteLine("fixedTrainPredictions from LinearRegression mse with multiplicativeCoeff=" + multiplicativeCoeff + ": " + ComputeMse(fixedTrainPredictions, id_to_info));
                var fixedValidationPredictions = ComputeFixedPredictions(validationPredictions, errorsLinearRegression, multiplicativeCoeff);
                Console.WriteLine("fixedValidationPredictions from LinearRegression mse with multiplicativeCoeff=" + multiplicativeCoeff + ": " + CFM60Utils.ComputeMse(fixedValidationPredictions));
            }

            testPredictions = CFM60Utils.LoadPredictions(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20211025\EnsembleLearning_637621601154307303_2x_network_actual_0_4069_add_ln_0then_addd_ln_1_12_actual_0_4050.csv");
            var errorsAvg = ComputeAverageError(trainPredictions);
            foreach (var multiplicativeCoeff in new[] { 1 })
            {
                //var fixedTrainPredictions = ComputeFixedPredictionsFromAverageError(trainPredictions, id_to_info, errorsAvg, multiplicativeCoeff);
                //Console.WriteLine("fixedTrainPredictions from Average mse with multiplicativeCoeff=" + multiplicativeCoeff + ": " + ComputeMse(fixedTrainPredictions, id_to_info));
                var fixedValidationPredictions = ComputeFixedPredictionsFromAverageError(validationPredictions, errorsAvg, multiplicativeCoeff);
                Console.WriteLine("fixedValidationPredictions from Average mse with multiplicativeCoeff=" + multiplicativeCoeff + ": " + CFM60Utils.ComputeMse(fixedValidationPredictions));
            }

            //var networkPredictions = LoadPredictions(@"C:\Temp\CFM60_30_0_2758_0_3748_20211024_2056_4.csv");
            testPredictions = CFM60Utils.LoadPredictions(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20211026\EnsembleLearning_637621601154307303_2x_network_actual_0_4069_add_ln_0then_addd_ln_1_12_actual_0_4050.csv");
            var networkErrorTestPredictions = CFM60Utils.LoadPredictions(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20211026\CFM60Errors_Encoder_1_30_0_16_0_2994_0_3534_20211026_1140_4_test.csv");
            foreach (var multiplicativeCoeff in new[] { 1 })
            {
                //var fixedTrainPredictions = ComputeFixedPredictionsFromAverageError(trainPredictions, id_to_info, errorsAvg, multiplicativeCoeff);
                //Console.WriteLine("fixedTrainPredictions from Average mse with multiplicativeCoeff=" + multiplicativeCoeff + ": " + ComputeMse(fixedTrainPredictions, id_to_info));
                var fixedValidationPredictions = ComputeFixedPredictionsFromDeepLearning(testPredictions, networkErrorTestPredictions, multiplicativeCoeff);
                Console.WriteLine("fixedValidationPredictions from DeepLearning with multiplicativeCoeff=" + multiplicativeCoeff + ": " + CFM60Utils.ComputeMse(fixedValidationPredictions));
                CFM60Utils.SavePredictions(fixedValidationPredictions, @"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20211026\EnsembleLearning_637621601154307303_2x_network_actual_0_4069_add_ln_0then_addd_ln_1_12_actual_0_4050_fixed.csv");
            }


            //NormalizePredictions(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20211027\EnsembleLearning_637621601154307303_2x_network_actual_0_4069_add_ln_0then_addd_ln_1_12_actual_0_4050.csv");
            NormalizePredictions(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20211027\EnsembleLearning_637621601154307303_2x_network_actual_0_4069_add_ln_0then_addd_ln_1_12_actual_0_4050_fixed_actual_0_4020.csv");
            return;


            foreach (var multiplicativeCoeff in new[] { 1 })
            {
                //var fixedTrainPredictions = ComputeFixedPredictionsFromAverageError(trainPredictions, id_to_info, errorsAvg, multiplicativeCoeff);
                //Console.WriteLine("fixedTrainPredictions from Average mse with multiplicativeCoeff=" + multiplicativeCoeff + ": " + ComputeMse(fixedTrainPredictions, id_to_info));
                var fixedValidationPredictions = ComputeFixedPredictionsFromDeepLearning(testPredictions, networkErrorTestPredictions, multiplicativeCoeff);
                Console.WriteLine("fixedValidationPredictions from DeepLearning with multiplicativeCoeff=" + multiplicativeCoeff + ": " + CFM60Utils.ComputeMse(fixedValidationPredictions));
                CFM60Utils.SavePredictions(fixedValidationPredictions, @"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20211026\EnsembleLearning_637621601154307303_2x_network_actual_0_4069_add_ln_0then_addd_ln_1_12_actual_0_4050_fixed.csv");
            }



            //foreach (var maxContiguousHolidays in new[] {3})
            //{
            //    var distances = HolidaysDistance(maxContiguousHolidays, x_train);
            //    string desc = " maxContiguousHolidays= " + maxContiguousHolidays;
            //    var distanceThresholds = new List<float> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20 };
            //    CreateFamilies(distances, distanceThresholds, desc, "c:/temp/holidaysFamily.csv");
            //}

            //var YCorrDistances = YCorrelationDistance(x_train);
            //Console.WriteLine("avg YCorrelationDistance="+(sum/YCorrDistances.Length));
            //var distanceThresholds = new List<float> { 0, 0.01f, 0.05f, 0.1f, 0.15f, 0.20f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
            //CreateFamilies(YCorrDistances, distanceThresholds, "", "c:/temp/correlationFamily.csv");
        }

        public static IDictionary<int, double> ComputeFixedPredictions(IDictionary<int, double> predictions, LinearRegression[] errorsLinearRegression, double multiplicativeCoeff)
        {
            var fixedPredictions = new Dictionary<int, double>();
            foreach (var p in predictions)
            {
                var id = p.Key;
                var prediction = p.Value;
                var idInfo = CFM60Utils.id_to_entries[id];
                var pid = idInfo.pid;
                var day = idInfo.day;
                var error = errorsLinearRegression[pid].Estimation(day);
                var fixedPrediction = prediction - multiplicativeCoeff * error;
                fixedPredictions[id] = fixedPrediction;
            }
            return fixedPredictions;
        }

        public static void NormalizePredictions(string filePath)
        {
            const double targetSumSquare = 3.0741775019772906;
            const double targetMean = -1.511705257;
            const double targetVariance = targetSumSquare - targetMean * targetMean;
            Console.WriteLine("target prediction E(Y^2)=" + targetSumSquare + " E(Y)=: " + targetMean + " VaR(Y)=: " + targetVariance);
            var testPredictions = CFM60Utils.LoadPredictions(filePath);
            var acc = new DoubleAccumulator();
            acc.Add(testPredictions.Values);
            var initialSumSquare = testPredictions.Values.Select(t => t * t).Sum() / testPredictions.Count;
            Console.WriteLine("original prediction E(Y^2)=" + initialSumSquare + " E(Y)=: " + acc.Average + " VaR(Y)=: " + acc.Variance);
            var fixedPred = new Dictionary<int, double>();
            var multCoeff = Math.Sqrt(targetVariance / acc.Variance);
            var addCoeff = targetMean - multCoeff * acc.Average;
            foreach (var e in testPredictions)
            {
                fixedPred[e.Key] = multCoeff * e.Value + addCoeff;
            }
            var fixedAcc = new DoubleAccumulator();
            fixedAcc.Add(fixedPred.Values);
            var fixedSumSquare = fixedPred.Values.Select(t => t * t).Sum() / fixedAcc.Count;
            Console.WriteLine("fixed prediction E(Y^2)=" + fixedSumSquare + " E(Y)=: " + fixedAcc.Average + " VaR(Y)=: " + fixedAcc.Variance);
            CFM60Utils.SavePredictions(fixedPred, filePath + "_normalized.csv");
        }

        public static IDictionary<int, double> ComputeFixedPredictionsFromAverageError(IDictionary<int, double> predictions, double[] averageErrors, double multiplicativeCoeff){
            var fixedPredictions = new Dictionary<int, double>();
            foreach (var p in predictions)
            {
                var id = p.Key;
                var prediction = p.Value;
                var entry = CFM60Utils.id_to_entries[id];
                var pid = entry.pid;
                var fixedPrediction = prediction - multiplicativeCoeff * averageErrors[pid];
                fixedPredictions[id] = fixedPrediction;
            }
            return fixedPredictions;
        }

        public static IDictionary<int, double> ComputeFixedPredictionsFromDeepLearning(IDictionary<int, double> predictions, IDictionary<int, double> networErrorskPredictions, double multiplicativeCoeff)
        {
            var fixedPredictions = new Dictionary<int, double>();
            foreach (var p in predictions)
            {
                var id = p.Key;
                var prediction = p.Value;
                var fixedPrediction = prediction + multiplicativeCoeff * networErrorskPredictions[id];
                fixedPredictions[id] = fixedPrediction;
            }
            return fixedPredictions;
        }

        public static LinearRegression[] ComputeLinearRegression(IDictionary<int, double> trainPredictions)
        {
            var res = new LinearRegression[pid_count];
            for (int i = 0; i < res.Length; ++i)
            {
                res[i] = new LinearRegression();
            }
            foreach (var p in trainPredictions)
            {
                var id = p.Key;
                var prediction = p.Value;
                var idInfo = CFM60Utils.id_to_entries[id];
                var pid = idInfo.pid;
                var day = idInfo.day;
                var Y = idInfo.Y;
                Debug.Assert(!float.IsNaN(Y));
                var error = prediction - Y;
                res[pid].Add(day, error);
            }
            return res;
        }

        public static double[] ComputeAverageError(IDictionary<int, double> trainPredictions)
        {
            var res = new double[pid_count];
            var resCount = new int[pid_count];
            foreach (var p in trainPredictions)
            {
                var id = p.Key;
                var prediction = p.Value;
                var idInfo = CFM60Utils.id_to_entries[id];
                var pid = idInfo.pid;
                var Y = idInfo.Y;
                Debug.Assert(!float.IsNaN(Y));
                var error = prediction - Y;
                res[pid] += error;
                ++resCount[pid];
            }
            for (int i = 0; i < res.Length; ++i)
                res[i] /= resCount[i];
            return res;
        }

     
        private static float[,] HolidaysDistance(int maxContiguousHolidays, List<CFM60Entry> entries)
        {
            var pid_to_working = new List<float?[]>();
            while (pid_to_working.Count < pid_count)
            {
                pid_to_working.Add(new float?[day_count]);
            }

            foreach (var e in entries)
            {
                var pid = e.pid;
                var day = e.day;
                pid_to_working[pid][day] = 1;
            }

            pid_to_working.ForEach(e => ReplaceSmallNanSequencesWith0(e, maxContiguousHolidays));
            var distances = new float[pid_count, pid_count];
            for (int pid0 = 0; pid0 < pid_count; ++pid0)
            for (int pid1 = 0; pid1 < pid_count; ++pid1)
                distances[pid0, pid1] = ComputeHolidaysDistance(pid_to_working[pid0], pid_to_working[pid1]);
            return distances;
        }

        private static float[,] YCorrelationDistance(List<CFM60Entry> entries)
        {
            var pid_to_Y = new List<float?[]>();
            while (pid_to_Y.Count < pid_count)
            {
                pid_to_Y.Add(new float?[day_count]);
            }
            foreach (var e in entries)
            {
                pid_to_Y[e.pid][e.day] = e.Y;
            }
            var distances = new float[pid_count, pid_count];
            for (int pid0 = 0; pid0 < pid_count; ++pid0)
            for (int pid1 = pid0; pid1 < pid_count; ++pid1)
            {
                distances[pid1, pid0] = distances[pid0, pid1] = ComputeYCorrelationDistance(pid_to_Y[pid0], pid_to_Y[pid1]);
            }
            return distances;
        }

        private static void ReplaceSmallNanSequencesWith0(float?[] a, int maxSmallSequenceSize)
        {
            for (int i = 0; i < a.Length; ++i)
            {
                if (a[i].HasValue)
                    continue;
                int nb = NbNaNStartingAtIndex(a, i);
                if (nb <= maxSmallSequenceSize)
                {
                    for (int j = i; j < i + nb; ++j)
                    {
                        a[j] = 0;
                    }
                }
                i += nb - 1;
            }
        }

        private static float ComputeYCorrelationDistance(float?[] a, float?[] b)
        {
            var lr = new LinearRegression();
            for (int i = 0; i < a.Length; ++i)
            {
                if (a[i].HasValue && b[i].HasValue)
                {
                    //lr.Add(a[i].Value, b[i].Value);
                    lr.Add(Math.Exp(a[i].Value), Math.Exp(b[i].Value));
                }
            }
            return 1.0f - (float)Math.Abs(lr.PearsonCorrelationCoefficient);
        }

        private static float ComputeHolidaysDistance(float?[] a, float?[] b)
        {
            float result = 0;
            for (int i = 0; i < a.Length; ++i)
            {
                if (a[i].HasValue && b[i].HasValue)
                {
                    result += Math.Abs(a[i].Value - b[i].Value);
                }
            }
            return result;
        }
        private static int NbNaNStartingAtIndex(float?[] a, int startIndex)
        {
            int result = 0;
            for (int i = startIndex; i < a.Length; ++i)
            {
                if (a[i].HasValue)
                    break;
                ++result;
            }
            return result;
        }



        private static void CreateFamilies(float[,] distances, List<float> distanceThresholds, string desc, string outputFile)
        {
            Debug.Assert(distances.GetLength(0) == distances.GetLength(1));
            int pid_count = distances.GetLength(0);
            var sb = new StringBuilder();
            sb.Append("Sep=;" + Environment.NewLine);
            sb.Append(";" + Environment.NewLine);
            var families = new List<List<int>>();
            for (int pid = 0; pid < pid_count; ++pid)
            {
                families.Add(new List<int> { pid });
            }
            foreach (var distanceThreshold in distanceThresholds)
            {
                families = families.OrderByDescending(f => f.Count).ToList();
                for (int family0 = 0; family0 < families.Count; ++family0)
                {
                    var f0 = families[family0];
                    for (int family1 = family0 + 1; family1 < families.Count; ++family1)
                    {
                        var f1 = families[family1];
                        if (MaxDistanceIsLessThenMaxAllowedDistance(f0, f1, distanceThreshold, distances))
                        {
                            f0.AddRange(f1);
                            families.RemoveAt(family1);
                            --family1;
                        }
                    }
                }

                families = families.OrderByDescending(f => f.Count).ToList();

                var orderedCount = families.Select(f => f.Count).ToList();
                var maxCount = orderedCount.Max();

                sb.Append(";" + Environment.NewLine);
                var msg = "with distanceThreshold=" + distanceThreshold + " " + desc + " => " + families.Count + " families" + " ( " + string.Join(" ", orderedCount) + " )";
                Console.WriteLine(msg);
                sb.Append(msg + Environment.NewLine);
                sb.Append(";" + Environment.NewLine);
                sb.Append("FamilyId;Count");
                for (int i = 0; i < maxCount; ++i)
                {
                    sb.Append(";pid" + i);
                }
                sb.Append(Environment.NewLine);

                int familyId = 0;
                foreach (var f in families.Where(f => f.Count >= 2))
                {
                    sb.Append(familyId + ";" + f.Count + ";" + string.Join(";", f) + Environment.NewLine);
                    familyId++;
                }

                var uniqueFamilies = families.Where(f => f.Count < 2).ToList();
                sb.Append("unique_family;" + uniqueFamilies.Count + ";" + string.Join(";", uniqueFamilies.SelectMany(x => x)) + Environment.NewLine);
            }
            File.WriteAllText(outputFile, sb.ToString());
        }


        private static bool MaxDistanceIsLessThenMaxAllowedDistance(List<int> family0, List<int> family1, float maxAllowedDistance, float[,] distances)
        {
            const float Epsilon = 1e-6f;

            foreach (var pid0 in family0)
                foreach (var pid1 in family1)
                {
                    if (distances[pid0, pid1] > (maxAllowedDistance + Epsilon))
                    {
                        return false;
                    }
                }
            return true;

        }


    }
}