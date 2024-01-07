using System.Collections.Generic;

namespace SharpNet.Datasets;

public abstract class WrappedDatasetSample : AbstractDatasetSample
{
    protected AbstractDatasetSample Original { get; }

    protected WrappedDatasetSample(AbstractDatasetSample original)
    {
        Original = original;
    }

    public override bool IsCategoricalColumn(string columnName) => Original.IsCategoricalColumn(columnName);
    public override string IdColumn => Original.IdColumn;
    public override string[] TargetLabels => Original.TargetLabels;
    public override Objective_enum GetObjective() => Original.GetObjective();
    public override DataFrame Predictions_InModelFormat_2_Predictions_InTargetFormat(DataFrame predictions_InModelFormat, Objective_enum objective) => Original.Predictions_InModelFormat_2_Predictions_InTargetFormat(predictions_InModelFormat, objective);
    public override int DatasetRows_InModelFormat_MustBeMultipleOf() => Original.DatasetRows_InModelFormat_MustBeMultipleOf();
    public override char GetSeparator() => Original.GetSeparator();
    public override void SavePredictions_InTargetFormat(DataFrame y_pred_Encoded_InTargetFormat, DataSet xDataset, string path) => Original.SavePredictions_InTargetFormat(y_pred_Encoded_InTargetFormat, xDataset, path);
    public override void SavePredictions_InModelFormat(DataFrame predictions_InModelFormat, string path) => Original.SavePredictions_InModelFormat(predictions_InModelFormat, path);
    public override HashSet<string> FieldsToDiscardInComputeHash() => Original.FieldsToDiscardInComputeHash();


}