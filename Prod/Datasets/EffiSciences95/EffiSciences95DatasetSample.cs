using System;
using System.Collections.Generic;
// ReSharper disable ConvertToConstant.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Datasets.EffiSciences95;

public class EffiSciences95DatasetSample : AbstractDatasetSample
{
    private EffiSciences95DirectoryDataSet _lazyFullTrainingAndValidation;
    private EffiSciences95DirectoryDataSet _lazyTestDataset;


    #region Hyper-Parameters
    public int MinEnlargeForBox = 0;
    public int MaxEnlargeForBox = 3;
    #endregion

    public EffiSciences95DatasetSample() : base(new HashSet<string>())
    {
    }
    public override string[] CategoricalFeatures => new string[0];
    public override string IdColumn => "index";
    public override string[] TargetLabels => new []{"labels"};
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }
    public override int[] GetInputShapeOfSingleElement()
    {
        return EffiSciences95Utils.Shape_CHW;
    }

    public override int[] X_Shape(int batchSize) => throw new NotImplementedException(); //!D TODO
    public override int[] Y_Shape(int batchSize) => throw new NotImplementedException(); //!D TODO

    public override int NumClass => TargetLabelDistinctValues.Length;
    public override string[] TargetLabelDistinctValues => new[] {"old" , "young"};

    public override DataSet TestDataset()
    {
        if (_lazyTestDataset == null || _lazyFullTrainingAndValidation.Disposed)
        {
            _lazyTestDataset = null;
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