using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.LightGBM;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Models;

public class ModelPredictions
{
    #region private fields
    /// <summary>
    /// if _header is true:
    ///     a line with the content of the header 
    /// else (_header is false):
    ///     null
    /// </summary>
    private readonly string _headerIfAny;
    private static readonly object LockSavePredictions = new();
    #endregion

    #region constructors
    private ModelPredictions(
        [NotNull] string trainPredictionsPath,
        [CanBeNull] string validationPredictionsPathIfAny,
        [CanBeNull] string testPredictionsPathIfAny,
        bool header,
        bool predictionsContainIndexColumn,
        char separator)
    {
        if (!File.Exists(trainPredictionsPath))
        {
            throw new ArgumentException($"file {trainPredictionsPath} is missing");
        }
        if (!string.IsNullOrEmpty(validationPredictionsPathIfAny) && !File.Exists(validationPredictionsPathIfAny))
        {
            throw new ArgumentException($"file {validationPredictionsPathIfAny} is missing");
        }
        if (!string.IsNullOrEmpty(testPredictionsPathIfAny) && !File.Exists(testPredictionsPathIfAny))
        {
            throw new ArgumentException($"file {testPredictionsPathIfAny} is missing");
        }
        TrainPredictionsPath = trainPredictionsPath;
        ValidationPredictionsPathIfAny = validationPredictionsPathIfAny;
        TestPredictionsPathIfAny = testPredictionsPathIfAny;
        Header = header;
        if (Header)
        {
            //TODO optimize this line: we only need to load the very first line of the file
            _headerIfAny = File.ReadLines(TrainPredictionsPath).First();
        }
        PredictionsContainIndexColumn = predictionsContainIndexColumn;
        Separator = separator;
    }
    public static ModelPredictions ValueOf(string workingDirectory, string modelName, bool header, bool predictionsContainIndexColumn, char separator)
    {
        var train = Directory.GetFiles(workingDirectory, modelName + "_predict_train*.csv");
        if (train.Length != 1)
        {
            throw new ArgumentException($"Fail to retrieve training predictions for {modelName} in directory {workingDirectory}");
        }
        var valid = Directory.GetFiles(workingDirectory, modelName + "_predict_valid*.csv");
        if (valid.Length >= 2)
        {
            throw new ArgumentException($"Too many validation predictions files {valid.Length} for {modelName} in directory {workingDirectory}");
        }
        var test = Directory.GetFiles(workingDirectory, modelName + "_predict_test*.csv");
        if (test.Length != 1)
        {
            throw new ArgumentException($"Too many test predictions files {test.Length} for {modelName} in directory {workingDirectory}");
        }
        return new ModelPredictions(
            train[0],
            valid.Length == 0 ? null : valid[0],
            test.Length == 0 ? null : test[0],
            header,
            predictionsContainIndexColumn,
            separator);
    }
    #endregion

    #region public fields
    [NotNull] public string TrainPredictionsPath { get; }
    [CanBeNull] public string ValidationPredictionsPathIfAny { get; }
    [CanBeNull] public string TestPredictionsPathIfAny { get; }
    public bool Header { get; }
    /// <summary>
    /// if true, the first columns of the model predictions contain the index of the row
    /// </summary>
    public bool PredictionsContainIndexColumn { get; }
    public char Separator { get; }
    #endregion

    public void SavePredictions(CpuTensor<float> predictions, string path)
    {
        StringBuilder sb = new();
        if (Header)
        {
            Debug.Assert(!string.IsNullOrEmpty(_headerIfAny));
            sb.Append(_headerIfAny + Environment.NewLine);
        }
        sb.Append(predictions.ToCsv(Separator, prefixWithRowIndex: PredictionsContainIndexColumn));
        var fileContent = sb.ToString();
        lock (LockSavePredictions)
        {
            File.WriteAllText(path, fileContent);
        }
    }
    [CanBeNull]  public CpuTensor<float> GetTrainPredictions()
    {
        return LoadPredictionsWithoutIndex(TrainPredictionsPath, Header, PredictionsContainIndexColumn, Separator);
    }
    [CanBeNull]  public CpuTensor<float> GetValidationPredictions()
    {
        return LoadPredictionsWithoutIndex(ValidationPredictionsPathIfAny, Header, PredictionsContainIndexColumn, Separator);
    }
    [CanBeNull] public CpuTensor<float> GetTestPredictions()
    {
        return LoadPredictionsWithoutIndex(TestPredictionsPathIfAny, Header, PredictionsContainIndexColumn, Separator);
    }

    private static CpuTensor<float> LoadPredictionsWithoutIndex(string path, bool header, bool predictionsContainIndexColumn, char separator)
    {
        if (string.IsNullOrEmpty(path))
        {
            return null;
        }
        var y_pred = Dataframe.Load(path, header, separator).Tensor;
        if (predictionsContainIndexColumn)
        {
            y_pred = y_pred.DropColumns(new[] { 0 });
        }
        return y_pred;
    }
}