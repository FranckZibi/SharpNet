using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Datasets;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public abstract class AbstractModelSample : AbstractSample, IMetricConfig
{
    protected AbstractModelSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(
        mandatoryCategoricalHyperParameters)
    {
    }

    public virtual IScore GetMinimumRankingScoreToSaveModel()
    {
        return null;
    }


    public bool IsRegressionProblem => GetObjective() == Objective_enum.Regression;


    public Objective_enum GetObjective()
    {
        switch (GetLoss())
        {

            case EvaluationMetricEnum.Accuracy:
            case EvaluationMetricEnum.SparseAccuracy:
            case EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy:
            case EvaluationMetricEnum.BinaryCrossentropy:
            case EvaluationMetricEnum.BCEContinuousY:
            case EvaluationMetricEnum.BCEWithFocalLoss:
            case EvaluationMetricEnum.CategoricalCrossentropy:
            case EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy:
            case EvaluationMetricEnum.SparseCategoricalCrossentropy:
            case EvaluationMetricEnum.F1Micro:
            case EvaluationMetricEnum.AUC:
            case EvaluationMetricEnum.AveragePrecisionScore:
                return Objective_enum.Classification;

            case EvaluationMetricEnum.Huber:
            case EvaluationMetricEnum.Mse:
            case EvaluationMetricEnum.MseOfLog:
            case EvaluationMetricEnum.MeanSquaredLogError:
            case EvaluationMetricEnum.Mae:
            case EvaluationMetricEnum.Rmse:
                return Objective_enum.Regression;

            case EvaluationMetricEnum.DEFAULT_VALUE:
                throw new ArgumentException($"loss function has not been set, please set it to a valid value and relaunch");
            default:
                throw new ArgumentException($"invalid loss function {GetLoss()}");
        }
    }

    #region IMetricConfig interface
    public abstract EvaluationMetricEnum GetLoss();
    public abstract EvaluationMetricEnum GetRankingEvaluationMetric();
    public List<EvaluationMetricEnum> Metrics
    {
        get
        {
            Debug.Assert(GetLoss() != EvaluationMetricEnum.DEFAULT_VALUE);
            var res = new List<EvaluationMetricEnum> { GetLoss() };
            foreach (var m in GetAllEvaluationMetrics())
            {
                if (m != GetLoss() && m != EvaluationMetricEnum.DEFAULT_VALUE)
                {
                    res.Add(m);
                }
            }
            return res;
        }
    }
    public virtual float Get_MseOfLog_Epsilon()
    {
        throw new NotImplementedException();
    }
    public virtual float Get_Huber_Delta()
    {
        throw new NotImplementedException();
    }
    public virtual float Get_BCEWithFocalLoss_PercentageInTrueClass()
    {
        throw new NotImplementedException();
    }

    public virtual float Get_BCEWithFocalLoss_Gamma()
    {
        throw new NotImplementedException();
    }

    #endregion

    protected abstract List<EvaluationMetricEnum> GetAllEvaluationMetrics();
    public abstract Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName);
    public abstract void Use_All_Available_Cores();

    public static AbstractModelSample LoadModelSample(string workingDirectory, string sampleName, bool useAllAvailableCores, Action<IDictionary<string, string>> contentUpdater = null)
    {
        var sample = (AbstractModelSample)ISample.Load(workingDirectory, sampleName, contentUpdater);
        if (useAllAvailableCores)
        {
            sample.Use_All_Available_Cores();
        }
        return sample;
    }
}