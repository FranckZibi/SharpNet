//using System.Collections.Generic;
//using SharpNet.CPU;
//using SharpNet.Datasets;
//using SharpNet.HyperParameters;
//using System;
//using SharpNet.Models;

//namespace SharpNet.HPO;

//public class WeightsOptimizerDatasetSample : AbstractDatasetSample
//{
//    #region private fields
//    private readonly AbstractDatasetSample _embeddedDatasetSample;
//    #endregion


//    public WeightsOptimizerDatasetSample(AbstractDatasetSample embeddedDatasetSample) : base(new HashSet<string>())
//    {
//        _embeddedDatasetSample = embeddedDatasetSample;
//        Test_DatasetPath = embeddedDatasetSample.Test_DatasetPath;
//        Train_XDatasetPath = embeddedDatasetSample.Train_XDatasetPath;
//        Train_YDatasetPath = embeddedDatasetSample.Train_YDatasetPath;
//        Validation_XDatasetPath = embeddedDatasetSample.Validation_XDatasetPath;
//        Validation_YDatasetPath = embeddedDatasetSample.Validation_YDatasetPath;
//    }

//    public override (string train_PredictionsFileName, float trainScore, string validation_PredictionsFileName, float validationScore, string test_PredictionsFileName)
//        Fit(IModel model, bool computeAndSavePredictions, bool computeValidationScore, bool saveTrainedModel)
//    {
//        var weightsOptimizer = (WeightsOptimizer)model;
//        var res = ("", float.NaN, "", float.NaN, "");
//        if (computeAndSavePredictions)
//        {
//            res = ComputeAndSavePredictions(model);
//        }
//        else if (computeValidationScore)
//        {
//            var validationScore = weightsOptimizer.Predictions().validationScore;
//            res = ("", float.NaN, "", validationScore, "");
//        }
//        if (saveTrainedModel)
//        {
//            model.Save(model.WorkingDirectory, model.ModelName);
//        }
//        return res;
//    }
//    public override (string train_PredictionsFileName, float trainScore, string validation_PredictionsFileName, float validationScore, string test_PredictionsFileName) ComputeAndSavePredictions(IModel model)
//    {
//        var weightsOptimizerModel = (WeightsOptimizer)model;
//        var (trainPredictions, trainScore, validationPredictions, validationScore, testPredictions, testScore) =
//            weightsOptimizerModel.Predictions();

//        Train_PredictionsFileName = "";
//        Validation_PredictionsFileName = "";
//        Test_PredictionsFileName = "";

//        if (!float.IsNaN(trainScore))
//        {
//            SaveTrainPredictions(model, trainPredictions, trainScore);
//        }
//        if (!float.IsNaN(validationScore))
//        {
//            SaveValidationPredictions(model, validationPredictions, validationScore);
//        }
//        if (testPredictions != null)
//        {
//            SaveTestPredictions(model, testPredictions);
//        }

//        return (Train_PredictionsFileName, trainScore, Validation_PredictionsFileName, validationScore, Test_PredictionsFileName);
//    }

//    public override List<string> CategoricalFeatures()
//    {
//        return _embeddedDatasetSample.CategoricalFeatures();
//    }

//    public override CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path)
//    {
//        return _embeddedDatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(dataframe_path);
//    }

//    public override (CpuTensor<float> trainPredictions, CpuTensor<float> validationPredictions, CpuTensor<float> testPredictions) LoadAllPredictions()
//    {
//        return _embeddedDatasetSample.LoadAllPredictions();
//    }


//    public override IDataSet TestDataset()
//    {
//        throw new NotImplementedException();
//    }

//    public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
//    {
//        throw new NotImplementedException();
//    }

//    public override (CpuTensor<float> trainPredictionsInTargetFormatWithoutIndex, CpuTensor<float> validationPredictionsInTargetFormatWithoutIndex,
//        CpuTensor<float> testPredictionsInTargetFormatWithoutIndex) LoadAllPredictionsInTargetFormatWithoutIndex()
//    {
//        throw new NotImplementedException();
//    }

//    public override CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(CpuTensor<float> predictionsInModelFormat)
//    {
//        throw new NotImplementedException();
//    }

//    public override CpuTensor<float> PredictionsInTargetFormat_2_PredictionsInModelFormat(CpuTensor<float> predictionsInTargetFormat)
//    {
//        throw new NotImplementedException();
//    }

//}