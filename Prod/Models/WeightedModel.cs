using System;
using System.Collections.Generic;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public class WeightedModel: AbstractModel
{
    #region private fields
    private readonly List<IModel> _embeddedModels;
    #endregion


    public WeightedModel(WeightedModelSample modelSample, string workingDirectory, string modelName) : base(modelSample, workingDirectory, modelName)
    {
        _embeddedModels = new();
        foreach(var (embeddedModelWorkingDirectory, embeddedModelName) in modelSample.GetWorkingDirectoryAndModelNames())
        {
            var embeddedModelSample = IModelSample.LoadModelSample(embeddedModelWorkingDirectory, embeddedModelName);
            _embeddedModels.Add(IModel.NewModel(embeddedModelSample, embeddedModelWorkingDirectory, embeddedModelName));
        }
    }

    public override (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath)
        Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny)
    {
        throw new System.NotImplementedException();
    }

    public override DataFrame Predict(IDataSet dataset, bool addIdColumnsAtLeft, bool removeAllTemporaryFilesAtEnd)
    {
        return WeightedModelSample.ApplyWeights(PredictForEachEmbeddedModel(dataset, addIdColumnsAtLeft),  new List<int>() );
    }

    private List<DataFrame> PredictForEachEmbeddedModel(IDataSet dataset, bool addIdColumnsAtLeft)
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

    public override int GetNumEpochs()
    {
        return _embeddedModels[0].GetNumEpochs();
    }

    public override string DeviceName()
    {
        return _embeddedModels[0].DeviceName();
    }
    public override double GetLearningRate()
    {
        return _embeddedModels[0].GetLearningRate();
    }
    public override void Use_All_Available_Cores()
    {
        _embeddedModels.ForEach(m => m.Use_All_Available_Cores());
    }

    public override int TotalParams()
    {
        return -1; //TODO
    }

    public override List<string> ModelFiles()
    {
        List<string> res = new();
        foreach (var m in _embeddedModels)
        {
            res.AddRange(m.ModelFiles());
        }
        return res;
    }

    private WeightedModelSample WeightedModelSample => (WeightedModelSample)ModelSample;
}
