using System;

namespace SharpNet.HyperParameters;

/// <summary>
/// a sample that has all needed information (model architecture + model HP + data) to train a model 
/// </summary>
public interface ITrainableSample: ISample
{
    IModelSample ModelSample { get; }
    AbstractDatasetSample DatasetSample { get; }

    public static ITrainableSample ValueOfITrainableSample(string workingDirectory, string modelName)
    {
        try { return TrainableSample.ValueOf(workingDirectory, modelName); }
        catch
        {
            // ignored
        }

        throw new Exception($"can't load trainable sample from model {modelName} in directory {workingDirectory}");
    }
}