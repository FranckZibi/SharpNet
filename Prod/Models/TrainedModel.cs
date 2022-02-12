using System;
using System.IO;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Datasets.Natixis70;
using SharpNet.HPO;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public class TrainedModel
{
    [NotNull] public string WorkingDirectory { get; }
    [NotNull] public string ModelName { get; }
    /// <summary>
    /// for a Neural Network model:
    ///     the weights associated with the model (h5 file format)
    /// for a LightGBM model:
    ///     the description of the trees embedded in the model (txt format)
    /// </summary>
    [CanBeNull] public string ModelDescriptionIfAny { get; }

    [NotNull] public ModelPredictions Predictions { get; }

    [NotNull] public ModelDatasets ModelDatasets { get; }

    /// <summary>
    /// Hyper-parameters used in the model
    /// </summary>
    [NotNull] public ISample Sample { get; }
    public ObjectiveFunction ObjectiveFunction { get; }



    public TrainedModel(
        [NotNull] string workingDirectory,
        [NotNull] string modelName,
        [CanBeNull] string modelDescriptionIfAny,
        [NotNull] ModelPredictions predictions,
        [NotNull] ModelDatasets modelDatasets,
        [NotNull] ISample sample,
        ObjectiveFunction objectiveFunction
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
        ObjectiveFunction = objectiveFunction;
        ModelDescriptionIfAny = modelDescriptionIfAny;
        Sample = sample;
    }


    public float ComputeLoss(CpuTensor<float> y_true, CpuTensor<float> y_predicted)
    {
        using var buffer = new CpuTensor<float>(new []{y_true.Shape[0]});
        switch (ObjectiveFunction)
        {
            case ObjectiveFunction.Rmse: return (float)y_true.ComputeRmse(y_predicted, buffer);
            case ObjectiveFunction.Mse: return (float)y_true.ComputeMse(y_predicted, buffer);
            default: throw new ArgumentException($"can not compute loss for {ObjectiveFunction}");
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
        if (sample is Natixis70_LightGBM_HyperParameters natixis70_LightGBM)
        {
            var natixis70DatasetHyperParameters = natixis70_LightGBM.DatasetHyperParameters;
            var lightGbmParameters = natixis70_LightGBM.LightGbmLightGbmSample;
            string xTestDatasetPath = natixis70DatasetHyperParameters.XTestDatasetPath();
            var modelDataset = lightGbmParameters.ToModelDatasets(xTestDatasetPath);
            var modelPredictions = ModelPredictions.ValueOf(workingDirectory, modelName, true, true, ',');
            return new TrainedModel(workingDirectory,  modelName, lightGbmParameters.output_model, modelPredictions, modelDataset, sample, ObjectiveFunction.Rmse);
        }

        if (sample is WeightsOptimizerHyperParameters weightsOptimizerSample)
        {
            var trainedModels = weightsOptimizerSample.LoadModelDescription(workingDirectory);
            var modelDescriptionPath = Path.Combine(workingDirectory, modelName + ".txt");
            var modelPredictions = ModelPredictions.ValueOf(workingDirectory, modelName,
                trainedModels[0].Predictions.Header,
                trainedModels[0].Predictions.PredictionsContainIndexColumn,
                trainedModels[0].Predictions.Separator);
            return new TrainedModel(workingDirectory, modelName, modelDescriptionPath, 
                modelPredictions,
                trainedModels[0].ModelDatasets, 
                sample,
                trainedModels[0].ObjectiveFunction);
        }

        throw new ArgumentException($"can't extract TrainedModel {modelName} from directory {workingDirectory}");
    }


}
