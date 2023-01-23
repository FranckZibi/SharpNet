using System.Collections.Generic;
// ReSharper disable ConvertToConstant.Global

namespace SharpNet.Datasets.EffiSciences95;

public class EffiSciences95DatasetSample : AbstractDatasetSample
{
    private EffiSciences95DirectoryDataSet _lazyFullTrainingAndValidation;
    private EffiSciences95DirectoryDataSet _lazyTestDataset;


    #region Hyper-Parameters
    public int MinEnlargeForBox = 0;
    public int MaxEnlargeForBox = 3;
    public bool EnlargeOldBoxToYoungBoxShape = true;
    public bool AddNewBoxOfOtherCategory = false;
    #endregion

    public EffiSciences95DatasetSample() : base(new HashSet<string>())
    {
    }
    public override string[] CategoricalFeatures => new string[0];
    public override string[] IdColumns => new[] { "index" };
    public override string[] TargetLabels => new []{"labels"};
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }
    public override int[] GetInputShapeOfSingleElement()
    {
        return EffiSciences95Utils.Shape_CHW;
    }
    public override int NumClass => 2;
    public override DataSet TestDataset()
    {
        if (_lazyTestDataset == null || _lazyFullTrainingAndValidation.Disposed)
        {
            _lazyTestDataset = null;
            //_LazyTestDataset = EffiSciences95DirectoryDataSet.ValueOf(false);
        }
        return _lazyTestDataset;
    }
    public override DataSet FullTrainingAndValidation()
    {
        if (_lazyFullTrainingAndValidation == null || _lazyFullTrainingAndValidation.Disposed)
        {
            _lazyFullTrainingAndValidation = EffiSciences95DirectoryDataSet.ValueOf(this, true);
        }

        return _lazyFullTrainingAndValidation;
    }
    public override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.Accuracy;
    }

    public override bool FixErrors()
    {
        if (!base.FixErrors())
        {
            return false;
        }

        if (MinEnlargeForBox > MaxEnlargeForBox)
        {
            return false;
        }

        if (!EnlargeOldBoxToYoungBoxShape && !AddNewBoxOfOtherCategory)
        {
            if (Utils.RandomCoinFlip())
            {
                EnlargeOldBoxToYoungBoxShape = true;
                AddNewBoxOfOtherCategory = false;
            }
            else
            {
                EnlargeOldBoxToYoungBoxShape = false;
                AddNewBoxOfOtherCategory = true;
            }
        }
        return true;
    }

    #region Dispose pattern
    protected override void Dispose(bool disposing)
    {
        if (disposed)
        {
            return;
        }
        disposed = true;
        //Release Unmanaged Resources
        if (disposing)
        {
            _lazyFullTrainingAndValidation?.Dispose();
            _lazyTestDataset?.Dispose();
            //Release Managed Resources
        }
    }
    #endregion
}