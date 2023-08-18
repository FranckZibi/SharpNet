using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.HyperParameters;
using SharpNet.MathTools;

namespace SharpNet;

public class EvaluationMetricAccumulatorForSingleEpoch : IDisposable
{
    private readonly TensorMemoryPool _memoryPool;
    private readonly int _count;
    private int _currentElementCount;
    private readonly IMetricConfig _metricData;
    private Tensor _full_y_true;
    private Tensor _full_y_pred;
    private readonly Dictionary<EvaluationMetricEnum, DoubleAccumulator> _currentAccumulatedMetrics = new();

    /// <summary>
    /// 
    /// </summary>
    /// <param name="memoryPool"></param>
    /// <param name="count">number of elements (rows) in the dataset</param>
    /// <param name="metricConfig"></param>
    public EvaluationMetricAccumulatorForSingleEpoch(TensorMemoryPool memoryPool, int count, IMetricConfig metricConfig)
    {
        _memoryPool = memoryPool;
        _count = count;
        _metricData = metricConfig;
    }

    public void UpdateMetrics([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted)
    {
        var remainingCount = _count - _currentElementCount;
        if (remainingCount <= 0) 
        {
            throw new ArgumentException($"remainingCount = {remainingCount} and received a tensor of {yExpected.Shape[0]} elements");
        }
        if (yExpected.Shape[0] > remainingCount)
        {
            var newShape = Utils.CloneShapeWithNewCount(yExpected.Shape, remainingCount);
            yExpected = yExpected.Reshape(newShape);
            yPredicted = yPredicted.Reshape(newShape);
        }

        if (_metricData.Metrics.Any(RequireFullYToCompute))
        {
            //we'll just update the tensors '_full_y_true' and '_full_y_pred' by appending to those tensors
            //the content of the tensors 'yExpected' and 'yPredicted'
            if (_full_y_true == null)
            {
                var full_y_true_shape = Utils.CloneShapeWithNewCount(yExpected.Shape, _count);
                _full_y_true = _memoryPool.GetFloatTensor(full_y_true_shape);
                _full_y_pred = _memoryPool.GetFloatTensor(_full_y_true.Shape);
            }
           
            Debug.Assert(yExpected.Shape[0] + _currentElementCount <= _count);
            yExpected.CopyTo(_full_y_true.Slice(_currentElementCount, yExpected.Shape));
            yPredicted.CopyTo(_full_y_pred.Slice(_currentElementCount, yPredicted.Shape));
        }
        else
        {
            Tensor buffer = null;
            foreach (var metric in _metricData.Metrics)
            {
                if (!_currentAccumulatedMetrics.ContainsKey(metric))
                {
                    _currentAccumulatedMetrics[metric] = new DoubleAccumulator();
                }
                _memoryPool.GetFloatTensor(ref buffer, yExpected.ComputeMetricBufferShape(metric));
                var metricValue = buffer.ComputeEvaluationMetric(yExpected, yPredicted, metric, _metricData);
                _currentAccumulatedMetrics[metric].Add(metricValue, yExpected.Shape[0]);
            }
            _memoryPool.FreeFloatTensor(buffer);
        }

        _currentElementCount += yExpected.Shape[0];
    }


    /// <summary>
    /// true if the metric can only be computed with the full y_true / y_false tensors
    /// </summary>
    /// <param name="metric"></param>
    /// <returns></returns>
    private static bool RequireFullYToCompute(EvaluationMetricEnum metric)
    {
        switch (metric)
        {
            case EvaluationMetricEnum.F1Micro:
            case EvaluationMetricEnum.PearsonCorrelation:
            case EvaluationMetricEnum.SpearmanCorrelation:
            case EvaluationMetricEnum.AUC:
                return true;
            default:
                return false;
        }
    }


    public List<KeyValuePair<EvaluationMetricEnum, double>> Metrics()
    {
        List<KeyValuePair<EvaluationMetricEnum, double>> res = new();

        if (_metricData.Metrics.Any(RequireFullYToCompute))
        {
            Debug.Assert(_full_y_true != null);
            Debug.Assert(_full_y_pred!=null);
            Tensor buffer = null;
            foreach (var metric in _metricData.Metrics)
            {
                _memoryPool.GetFloatTensor(ref buffer, _full_y_pred.ComputeMetricBufferShape(metric));
                res.Add(KeyValuePair.Create(metric, buffer.ComputeEvaluationMetric(_full_y_true, _full_y_pred, metric, _metricData)));
            }
            _memoryPool.FreeFloatTensor(buffer);
        }
        else
        {
            foreach (var metric in _metricData.Metrics)
            {
                res.Add(KeyValuePair.Create(metric, _currentAccumulatedMetrics[metric].Average));
            }
        }
        return res;
    }

    public void Dispose()
    {
        _full_y_true?.Dispose();
        _full_y_pred?.Dispose();
        _currentAccumulatedMetrics.Clear();
    }
}