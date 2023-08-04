using System.Collections.Generic;

namespace SharpNet.Datasets;

public abstract class WrappedDatasetSample : AbstractDatasetSample
{
    protected AbstractDatasetSample Original { get; }

    protected WrappedDatasetSample(AbstractDatasetSample original) : base(new HashSet<string>())
    {
        Original = original;
    }

    public override string[] CategoricalFeatures => Original.CategoricalFeatures;
    public override string IdColumn => Original.IdColumn;
    public override string[] TargetLabels => Original.TargetLabels;
    public override string[] TargetLabelDistinctValues => Original.TargetLabelDistinctValues;
    public override Objective_enum GetObjective() => Original.GetObjective();
    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat, Objective_enum objective) => Original.PredictionsInModelFormat_2_PredictionsInTargetFormat(predictionsInModelFormat, objective);
    public override int DatasetRowsInModelFormatMustBeMultipleOf() => Original.DatasetRowsInModelFormatMustBeMultipleOf();
    public override char GetSeparator() => Original.GetSeparator();
    public override void SavePredictionsInTargetFormat(DataFrame y_pred_Encoded_InTargetFormat, DataSet xDataset, string path) => Original.SavePredictionsInTargetFormat(y_pred_Encoded_InTargetFormat, xDataset, path);
    public override void SavePredictionsInModelFormat(DataFrame predictionsInModelFormat, string path) => Original.SavePredictionsInModelFormat(predictionsInModelFormat, path);
    public override HashSet<string> FieldsToDiscardInComputeHash() => Original.FieldsToDiscardInComputeHash();


}