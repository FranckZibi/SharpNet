using System;

namespace SharpNet.Datasets;

public class ConstDatasetSample : AbstractDatasetSample
{
    private readonly Objective_enum Objective;
    private readonly int[] x_shape_for_1_batchSize;
    private readonly int[] y_shape_for_1_batchSize;
    private readonly Func<string, bool> isCategoricalColumn;

    public ConstDatasetSample(string idColumn, string[] targetLabels, int[] x_shape_for_1_batchSize,
        int[] y_shape_for_1_batchSize, int numClass, Objective_enum objective, Func<string, bool> isCategoricalColumn)
    {
        Objective = objective;
        IdColumn = idColumn;
        TargetLabels = targetLabels;
        this.x_shape_for_1_batchSize = x_shape_for_1_batchSize;
        this.y_shape_for_1_batchSize = y_shape_for_1_batchSize;
        NumClass = numClass;
        this.isCategoricalColumn = isCategoricalColumn;
    }


    public override string IdColumn { get; }
    public override string[] TargetLabels { get; }

    public override bool IsCategoricalColumn(string columnName)
    {
        if (isCategoricalColumn != null)
        {
            return isCategoricalColumn(columnName);
        }

        return IsIdColumn(columnName);
    }
    public override int[] X_Shape(int batchSize)
    {
        var res = (int[])x_shape_for_1_batchSize.Clone();
        res[0] = batchSize;
        return res;
    }
    public override int[] Y_Shape(int batchSize) => Utils.CloneShapeWithNewCount(y_shape_for_1_batchSize, batchSize);

    public override int NumClass { get; }
    public override Objective_enum GetObjective() => Objective;

    public override DataSet TestDataset()
    {
        throw new NotImplementedException();
    }

    public override DataSet FullTrainingAndValidation()
    {
        throw new NotImplementedException();
    }
}