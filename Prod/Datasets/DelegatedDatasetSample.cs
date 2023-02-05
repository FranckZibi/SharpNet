using System.Collections.Generic;

namespace SharpNet.Datasets;

public abstract class DelegatedDatasetSample : AbstractDatasetSample
{
    protected AbstractDatasetSample EmbeddedDatasetSample { get; }

    protected DelegatedDatasetSample(AbstractDatasetSample embeddedDatasetSample) : base(new HashSet<string>())
    {
        EmbeddedDatasetSample = embeddedDatasetSample;
    }

    public override string[] CategoricalFeatures => EmbeddedDatasetSample.CategoricalFeatures;
    public override string[] IdColumns => EmbeddedDatasetSample.IdColumns;
    public override string[] TargetLabels => EmbeddedDatasetSample.TargetLabels;
    public override string[] TargetLabelDistinctValues => EmbeddedDatasetSample.TargetLabelDistinctValues;
    public override Objective_enum GetObjective() => EmbeddedDatasetSample.GetObjective();
    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat) => EmbeddedDatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(predictionsInModelFormat);
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => EmbeddedDatasetSample.GetRankingEvaluationMetric();
    public override int DatasetRowsInModelFormatMustBeMultipleOf() => EmbeddedDatasetSample.DatasetRowsInModelFormatMustBeMultipleOf();
    public override char GetSeparator() => EmbeddedDatasetSample.GetSeparator();
    public override void SavePredictionsInTargetFormat(DataFrame y_pred_Encoded_InTargetFormat, DataSet xDataset, string path) => EmbeddedDatasetSample.SavePredictionsInTargetFormat(y_pred_Encoded_InTargetFormat, xDataset, path);
    public override void SavePredictionsInModelFormat(DataFrame predictionsInModelFormat, string path) => EmbeddedDatasetSample.SavePredictionsInModelFormat(predictionsInModelFormat, path);
    //public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation() => EmbeddedDatasetSample.SplitIntoTrainingAndValidation();
    public override IScore MinimumScoreToSaveModel => EmbeddedDatasetSample.MinimumScoreToSaveModel;
    public override HashSet<string> FieldsToDiscardInComputeHash() => EmbeddedDatasetSample.FieldsToDiscardInComputeHash();


}