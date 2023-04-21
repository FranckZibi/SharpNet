using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.MathTools;

namespace SharpNet;

public class EvaluationMetricAccumulator
{
    private readonly TensorMemoryPool _memoryPool;

    private readonly Dictionary<EvaluationMetricEnum, DoubleAccumulator> _currentAccumulatedMetrics = new();


    public EvaluationMetricAccumulator(TensorMemoryPool MemoryPool)
    {
        _memoryPool = MemoryPool;
    }

    public void UpdateMetrics([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted, List<EvaluationMetricEnum> metrics)
    {
        foreach (var metric in metrics)
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
                var metricValue = yExpected.ComputeEvaluationMetric(yPredicted, metric, buffer);
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