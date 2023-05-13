using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.MathTools;

namespace SharpNet;

public class EvaluationMetricAccumulatorForSingleEpoch
{
    private readonly TensorMemoryPool _memoryPool;
    private readonly int _count;
    private readonly List<EvaluationMetricEnum> _metrics;

    private readonly Dictionary<EvaluationMetricEnum, DoubleAccumulator> _currentAccumulatedMetrics = new();

    /// <summary>
    /// 
    /// </summary>
    /// <param name="MemoryPool"></param>
    /// <param name="count">number of elements (rows) in the dataset</param>
    /// <param name="metrics"></param>
    public EvaluationMetricAccumulatorForSingleEpoch(TensorMemoryPool MemoryPool, int count, List<EvaluationMetricEnum> metrics)
    {
        _memoryPool = MemoryPool;
        _count = count;
        _metrics = metrics;
    }

    public void UpdateMetrics([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted)
    {
        foreach (var metric in _metrics)
        {
            var buffer= _memoryPool.GetFloatTensor(yExpected.ComputeMetricBufferShape(metric));
            UpdateMetric(yExpected, yPredicted, metric, buffer);
            _memoryPool.FreeFloatTensor(buffer);
        }
    }

    private void UpdateMetric([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted, EvaluationMetricEnum metric, Tensor buffer)
    {
        switch (metric)
        {
            case EvaluationMetricEnum.F1Micro:
            case EvaluationMetricEnum.PearsonCorrelation:
            case EvaluationMetricEnum.SpearmanCorrelation:
                throw new ArgumentException($"EvaluationMetricEnum.{metric} is not supported");
            default:
                if (!_currentAccumulatedMetrics.ContainsKey(metric))
                {
                    _currentAccumulatedMetrics[metric] = new DoubleAccumulator();
                }
                var metricValue = buffer.ComputeEvaluationMetric(yExpected, yPredicted, metric);
                _currentAccumulatedMetrics[metric].Add(metricValue, yExpected.Shape[0]);
                break;
        }
    }


    public Dictionary<EvaluationMetricEnum, double> Metrics()
    {
        Dictionary<EvaluationMetricEnum, double> res = new();
        foreach (var (m, acc) in _currentAccumulatedMetrics)
        {
            res[m] = acc.Average; ;
        }
        return res;
    }
}