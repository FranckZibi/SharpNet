using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpNet.Networks
{
    public partial class Network
    {
        #region private fields
        private readonly Stopwatch _spInternalFit = new ();
        private readonly Stopwatch _swComputeMetrics;
        private readonly IDictionary<string, Stopwatch> _updateWeightsTime = new Dictionary<string, Stopwatch>();
        #endregion

        public IDictionary<string, Stopwatch> ForwardPropagationTrainingTime { get; } = new Dictionary<string, Stopwatch>();
        public IDictionary<string, Stopwatch> ForwardPropagationInferenceTime { get; } = new Dictionary<string, Stopwatch>();
        public IDictionary<string, Stopwatch> BackwardPropagationTime { get; } = new Dictionary<string, Stopwatch>();

        private string LayersKpi()
        {
            var totalSeconds = _spInternalFit.Elapsed.TotalSeconds;
            var result = "Took " + Math.Round(totalSeconds, 1) + "s";
            result += " (Metrics:" + Math.Round(100 * _swComputeMetrics.Elapsed.TotalSeconds / totalSeconds, 0) + "%)" + Environment.NewLine;
            result += KpiByLayerType(totalSeconds);
            return result;
        }
        private string KpiByLayerType(double totalSeconds)
        {
            double PercentageOfTimeTaken(IDictionary<string, Stopwatch> layerTypeToTimer, string layerType)
            {
                return layerTypeToTimer.TryGetValue(layerType, out var sw) ? (sw.Elapsed.TotalSeconds / Math.Max(totalSeconds, 1e-6)) : 0.0;
            }
            string ParentLayerName(string keyName)
            {
                int idx = keyName.IndexOf(">", StringComparison.Ordinal);
                return (idx > 0) ? keyName.Substring(0, idx) : keyName;
            }
            double ParentTime(string keyName, List<Tuple<string, double, double, double, double>> values)
            {
                var parent = ParentLayerName(keyName);
                var parentTuple = values.FirstOrDefault(t => t.Item1 == parent);
                return parentTuple == null ? 0 : parentTuple.Item2 + parentTuple.Item3 + parentTuple.Item4 + parentTuple.Item5;
            }
            var data = new List<Tuple<string, double, double, double, double>>();
            var separatingLine = new string('=', 100);
            var allKeys = ForwardPropagationTrainingTime.Keys.Union(BackwardPropagationTime.Keys).Union(ForwardPropagationInferenceTime.Keys).Union(_updateWeightsTime.Keys).ToList();
            foreach (var layerType in allKeys)
            {
                data.Add(Tuple.Create(layerType, PercentageOfTimeTaken(ForwardPropagationTrainingTime, layerType), PercentageOfTimeTaken(BackwardPropagationTime, layerType), PercentageOfTimeTaken(ForwardPropagationInferenceTime, layerType), PercentageOfTimeTaken(_updateWeightsTime, layerType)));
            }

            data = data.OrderByDescending(t => ParentTime(t.Item1, data)).ThenBy(t => t.Item1).ToList();
            var result = separatingLine + Environment.NewLine;
            result += "LayerName              Forward(Training)  Backward(Training)  Forward(Inference)        UpdateHeight" + Environment.NewLine;
            result += separatingLine + Environment.NewLine;
            result += string.Join(Environment.NewLine, data.Select(d => KpiByLayerTypeSingleLine(d.Item1, d.Item2, d.Item3, d.Item4, d.Item5))) + Environment.NewLine;
            //we compute the total by column
            result += separatingLine + Environment.NewLine;
            var dataWithoutDuplicate = data.Where(t => !t.Item1.Contains(">")).ToList();
            result += KpiByLayerTypeSingleLine("", dataWithoutDuplicate.Select(t => t.Item2).Sum(), dataWithoutDuplicate.Select(t => t.Item3).Sum(), dataWithoutDuplicate.Select(t => t.Item4).Sum(), dataWithoutDuplicate.Select(t => t.Item5).Sum()) + Environment.NewLine;
            result += separatingLine + Environment.NewLine;
            return result;
        }
        private static string KpiByLayerTypeSingleLine(string layerType, double forwardPropagationTraining, double forwardPropagationInference, double backwardPropagation, double totalUpdateWeights)
        {
            string AsDisplayString(double d)
            {
                return Math.Round(d * 100, 1) + "%";
            }
            string SubCategoryLayerName(string keyName)
            {
                int idx = keyName.IndexOf(">", StringComparison.Ordinal);
                return (idx >= 0) ? keyName.Substring(idx) : "";
            }
            const int columnWidth = 20;
            return (layerType.Contains(">") ? $"{SubCategoryLayerName(layerType),20}" : $"{layerType,-20}")
                   + $"{AsDisplayString(forwardPropagationTraining),columnWidth}{AsDisplayString(forwardPropagationInference),columnWidth}{AsDisplayString(backwardPropagation),columnWidth}{AsDisplayString(totalUpdateWeights),columnWidth}".TrimEnd();
        }
        public static void StartTimer(string key, IDictionary<string, Stopwatch> layerTypeToStopWatch)
        {
            if (layerTypeToStopWatch.TryGetValue(key, out Stopwatch sw))
            {
                sw.Start();
                return;
            }
            layerTypeToStopWatch[key] = Stopwatch.StartNew();
        }
        public static void StopTimer(string key, IDictionary<string, Stopwatch> layerTypeToStopWatch)
        {
            Debug.Assert(layerTypeToStopWatch.ContainsKey(key));
            layerTypeToStopWatch[key].Stop();
        }
    }
}