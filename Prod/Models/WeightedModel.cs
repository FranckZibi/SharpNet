using System;
using System.Collections.Generic;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public class WeightedModel: Model
{
    #region private fields
    private readonly List<Model> _embeddedModels;
    #endregion


    public WeightedModel(WeightedModelSample modelSample, string workingDirectory, string modelName) : base(modelSample, workingDirectory, modelName)
    {
        _embeddedModels = new();
        foreach(var (embeddedModelWorkingDirectory, embeddedModelName) in modelSample.GetWorkingDirectoryAndModelNames())
        {
            var embeddedModelSample = IModelSample.LoadModelSample(embeddedModelWorkingDirectory, embeddedModelName);
            _embeddedModels.Add(NewModel(embeddedModelSample, embeddedModelWorkingDirectory, embeddedModelName));
        }
    }

    public override (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath)
        Fit(DataSet trainDataset, DataSet validationDatasetIfAny)
    {
        throw new System.NotImplementedException();
    }

    public override DataFrame Predict(DataSet dataset, bool addIdColumnsAtLeft, bool removeAllTemporaryFilesAtEnd)
    {
        return WeightedModelSample.ApplyWeights(PredictForEachEmbeddedModel(dataset, addIdColumnsAtLeft),  new List<int>() );
    }

    private List<DataFrame> PredictForEachEmbeddedModel(DataSet dataset, bool addIdColumnsAtLeft)
    {
        var allModelPredictions = new List<DataFrame>();
        foreach (var m in _embeddedModels)
        {
            var modelPrediction = m.Predict(dataset, addIdColumnsAtLeft, false);
            allModelPredictions.Add(modelPrediction);
        }
        return allModelPredictions;
    }

    public override void Save(string workingDirectory, string modelName)
    {
        //no need to save the embedded models : they are readonly
        ModelSample.Save(workingDirectory, modelName);
    }


    public override List<string> ModelFiles() => new();

    private WeightedModelSample WeightedModelSample => (WeightedModelSample)ModelSample;
}
