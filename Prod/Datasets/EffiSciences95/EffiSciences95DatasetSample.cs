using System;
using System.Collections.Generic;
// ReSharper disable ConvertToConstant.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Datasets.EffiSciences95;

/// <summary>
/// this class contains the hyper parameters related to the dataset (% of data in training/validation; KFold, shuffling, parameters related to the box label removal, etc...)
/// </summary>
public class EffiSciences95DatasetSample : AbstractDatasetSample
{
    private EffiSciences95DirectoryDataSet LazyFullTrainingAndValidation;
    private EffiSciences95DirectoryDataSet LazyTestDataset;


    #region Hyper-Parameters
    /// <summary>
    /// when removing the box containing the label, we enlarge this box by a random value between MinEnlargeForBox and MaxEnlargeForBox on each 4 sides of the label box
    /// </summary>
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
        if (LazyTestDataset == null || LazyFullTrainingAndValidation.Disposed)
        {
            LazyTestDataset = null;
        }
        return LazyTestDataset;
    }
    public override DataSet FullTrainingAndValidation()
    {
        if (LazyFullTrainingAndValidation == null || LazyFullTrainingAndValidation.Disposed)
        {
            LazyFullTrainingAndValidation = EffiSciences95DirectoryDataSet.ValueOf(this, "Labeled");
        }

        return LazyFullTrainingAndValidation;
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
            LazyFullTrainingAndValidation?.Dispose();
            LazyTestDataset?.Dispose();
            //Release Managed Resources
        }
    }
    #endregion
}