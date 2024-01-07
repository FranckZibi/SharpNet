using System;

// ReSharper disable ConvertToConstant.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Datasets.EffiSciences95;

/// <summary>
/// this class contains the Hyperparameters related to the dataset (% of data in training/validation; KFold, shuffling, parameters related to the box label removal, etc...)
/// </summary>
public class EffiSciences95DatasetSample : AbstractDatasetSample
{
    private EffiSciences95DirectoryDataSet LazyFullTrainingAndValidation;
    private EffiSciences95DirectoryDataSet LazyTestDataset;


    #region Hyperparameters
    /// <summary>
    /// when removing the box containing the label, we enlarge this box by a random value between MinEnlargeForBox and MaxEnlargeForBox on each 4 sides of the label box
    /// </summary>
    public int MinEnlargeForBox = 0;
    public int MaxEnlargeForBox = 3;
    #endregion

    // ReSharper disable once EmptyConstructor
    public EffiSciences95DatasetSample()
    {
    }
    public override string IdColumn => "index";
    public override string[] TargetLabels => new []{"labels"};
    public override bool IsCategoricalColumn(string columnName) => DefaultIsCategoricalColumn(columnName);


    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }
    public override int[] X_Shape(int batchSize)
    {
        return new []{ batchSize,  3, 218, 178 };
    }
    public override int[] Y_Shape(int batchSize) => throw new NotImplementedException(); //!D TODO

    public override int NumClass => EffiSciences95Utils.TargetLabelDistinctValues.Length;

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