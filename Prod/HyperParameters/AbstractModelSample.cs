using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Datasets;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public abstract class AbstractModelSample : AbstractSample, IModelSample
{
    protected AbstractModelSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(
        mandatoryCategoricalHyperParameters)
    {
    }

    public abstract EvaluationMetricEnum GetLoss();

    public bool IsRegressionProblem => GetObjective() == Objective_enum.Regression;

    public Objective_enum GetObjective()
    {
        switch (GetLoss())
        {

            case EvaluationMetricEnum.Accuracy:
            case EvaluationMetricEnum.SparseAccuracy:
            case EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy:
            case EvaluationMetricEnum.BinaryCrossentropy:
            case EvaluationMetricEnum.CategoricalCrossentropy:
            case EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy:
            case EvaluationMetricEnum.SparseCategoricalCrossentropy:
            case EvaluationMetricEnum.F1Micro:
            case EvaluationMetricEnum.AUC:
                return Objective_enum.Classification;

            case EvaluationMetricEnum.Huber:
            case EvaluationMetricEnum.Mse:
            case EvaluationMetricEnum.MseOfLog:
            case EvaluationMetricEnum.MeanSquaredLogError:
            case EvaluationMetricEnum.Mae:
            case EvaluationMetricEnum.Rmse:
                return Objective_enum.Regression;
            default:
                throw new ArgumentException($"invalid loss function {GetLoss()}");
        }
    }

    public List<EvaluationMetricEnum> Metrics
    {
        get
        {
            Debug.Assert(GetLoss() != EvaluationMetricEnum.DEFAULT_VALUE);
            var res = new List<EvaluationMetricEnum> { GetLoss() };
            if (GetRankingEvaluationMetric() != GetLoss() && GetRankingEvaluationMetric() != EvaluationMetricEnum.DEFAULT_VALUE)
            {
                res.Add(GetRankingEvaluationMetric());
            }
            return res;
        }
    }

    public abstract EvaluationMetricEnum GetRankingEvaluationMetric();
    public abstract Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName);
    public abstract void Use_All_Available_Cores();

    public static AbstractModelSample LoadModelSample(string workingDirectory, string sampleName, bool useAllAvailableCores)
    {
        var sample = (AbstractModelSample)ISample.Load(workingDirectory, sampleName);
        if (useAllAvailableCores)
        {
            sample.Use_All_Available_Cores();
        }
        return sample;
    }
}