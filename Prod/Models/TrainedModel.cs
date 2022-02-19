using System;
using System.IO;
using JetBrains.Annotations;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public class TrainedModel
{
    private readonly MetricEnum _metricFunction;
    [NotNull] public string WorkingDirectory { get; }
    [NotNull] public string ModelName { get; }
    /// <summary>
    /// for a Neural Network model:
    ///     the weights associated with the model (h5 file format)
    /// for a LightGBM model:
    ///     the description of the trees embedded in the model (txt format)
    /// </summary>
    //[CanBeNull] private string ModelDescriptionIfAny { get; }

    [NotNull] public ModelPredictions Predictions { get; }

    [NotNull] private ModelDatasets ModelDatasets { get; }

    /// <summary>
    /// Hyper-parameters used in the model
    /// </summary>
    [NotNull]  private ISample Sample { get; }

    private MetricEnum MetricEnum()
    {
        return _metricFunction;
    }


    private TrainedModel(
        [NotNull] string workingDirectory,
        [NotNull] string modelName,
        [NotNull] ModelPredictions predictions,
        ModelDatasets modelDatasets,
        [NotNull] ISample sample,
        MetricEnum metricFunction
        )
    {
        if (!Directory.Exists(workingDirectory))
        {
            throw new Exception($"invalid directory: {workingDirectory}");
        }

        WorkingDirectory = workingDirectory;
        ModelName = modelName;
        Predictions = predictions;
        ModelDatasets = modelDatasets;
        _metricFunction = metricFunction;
        Sample = sample;
    }


    public float ComputeLoss(CpuTensor<float> y_true, CpuTensor<float> y_predicted)
    {
        using var buffer = new CpuTensor<float>(new []{y_true.Shape[0]});
        switch (MetricEnum())
        {
            case SharpNet.MetricEnum.Rmse: return (float)y_true.ComputeRmse(y_predicted, buffer);
            case SharpNet.MetricEnum.Mse: return (float)y_true.ComputeMse(y_predicted, buffer);
            default: throw new ArgumentException($"can not compute loss for {MetricEnum()}");
        }
    }

    public CpuTensor<float> Perfect_Train_Predictions_if_any()
    {
        return string.IsNullOrEmpty(ModelDatasets.Y_train_dataset_path)
            ? null
            : Sample.Y_Train_dataset_to_Perfect_Predictions(ModelDatasets.Y_train_dataset_path);
    }
    public CpuTensor<float> Perfect_Validation_Predictions_if_any()
    {
        return string.IsNullOrEmpty(ModelDatasets.Y_validation_dataset_path)
            ? null
            : Sample.Y_Train_dataset_to_Perfect_Predictions(ModelDatasets.Y_validation_dataset_path);
    }

    public static TrainedModel ValueOf([NotNull] string workingDirectory, [NotNull] string modelName)
    {
        var sample = ISample.ValueOf(workingDirectory, modelName);
        if (sample is Model_and_Dataset_Sample modelAndDatasetSample)
        {
            var datasetSample = modelAndDatasetSample.DatasetSample;
            var modelDataset = datasetSample.ToModelDatasets();
            var modelPredictions = ModelPredictions.ValueOf(workingDirectory, modelName, true, true, ',');
            return new TrainedModel(workingDirectory,  modelName, modelPredictions, modelDataset, sample, SharpNet.MetricEnum.Rmse);
        }
        if (sample is WeightsOptimizerHyperParameters weightsOptimizerSample)
        {
            var firstTrainedModels = weightsOptimizerSample.LoadModelDescription(workingDirectory)[0];
            var modelPredictions = ModelPredictions.ValueOf(workingDirectory, modelName,
                firstTrainedModels.Predictions.Header,
                firstTrainedModels.Predictions.PredictionsContainIndexColumn,
                firstTrainedModels.Predictions.Separator);
            return new TrainedModel(
                workingDirectory, 
                modelName, 
                modelPredictions,
                firstTrainedModels.ModelDatasets,
                sample,
                firstTrainedModels.MetricEnum());
        }

        if (sample is KFoldSample kfoldSample)
        {
            var modelPredictions = ModelPredictions.ValueOf(workingDirectory, modelName, true, true, ',');
            return new TrainedModel(workingDirectory, modelName, modelPredictions, null, sample, kfoldSample.Metric);
        }
        if (sample is CatBoostSample)
        {
            //var natixis70DatasetHyperParameters = natixis70_CatBoost.DatasetHyperParameters;
            //string xTestDatasetPath = natixis70DatasetHyperParameters.XTestDatasetPath();
            //var modelDataset = catBoostSample.ToModelDatasets(xTestDatasetPath);
            var modelPredictions = ModelPredictions.ValueOf(workingDirectory, modelName, true, true, ',');
            return new TrainedModel(workingDirectory, modelName, modelPredictions, null, sample, SharpNet.MetricEnum.Rmse);
        }

        throw new ArgumentException($"can't extract TrainedModel {modelName} from directory {workingDirectory}");
    }


}
