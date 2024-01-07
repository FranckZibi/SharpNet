using log4net;
using System;
using System.Diagnostics;
using System.Linq;
// ReSharper disable FieldCanBeMadeReadOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.PlumeLabs88;

public class PlumeLabs88DatasetSample : AbstractDatasetSample
{
    #region private fields & properties
    // ReSharper disable once UnusedMember.Local
    private static readonly ILog Log = LogManager.GetLogger(typeof(PlumeLabs88DatasetSample));
    #endregion

    #region Hyperparameters
    // ReSharper disable once UnusedMember.Global
    public bool fillna_with_0 = false;
    public float NormalizeFeatureMean = 0.0f;
    public float NormalizeFeatureMult = 1.0f;
    public float NormalizeTargetMean = 0.0f;
    public float NormalizeTargetMult = 1.0f;
    /// <summary>
    /// the target shape fo the features
    /// it must be among:  (2, 4, 8, 16, 32, 64, 128)
    /// </summary>
    public int TargetHeightAndWidth = 128;
    public int MaxIdTraining = PlumeLabs88Utils.RawMaxIdTraining;
    public int MaxIdTest = PlumeLabs88Utils.RawMaxIdTest;
    #endregion




    // ReSharper disable once EmptyConstructor
    public PlumeLabs88DatasetSample()
    {
    }

    public override int[] X_Shape(int batchSize)
    {
        int[] res = new[]{batchSize}.Concat(PlumeLabs88Utils.Raw_Shape_CHW).ToArray();
        res[2] = res[3] = TargetHeightAndWidth;
        return res;
    }
    public override int[] Y_Shape(int batchSize) => throw new NotImplementedException(); //!D TODO
    public override int NumClass => 8;

    public int DatasetMaxId(bool isTrainingDataset) { return isTrainingDataset ? MaxIdTraining : MaxIdTest; }

    public override string IdColumn { get; } = PlumeLabs88Utils.ID_COLUMN_NAME;
    public override string[] TargetLabels { get; } = { "TARGET" };
    public override bool IsCategoricalColumn(string columnName) { return Equals(columnName, IdColumn); }

    public override Objective_enum GetObjective()
    {
        return Objective_enum.Regression;
    }




    public string[] RowInTargetFormatPredictionToID(bool isTrainingDataset)
    {
        var df = isTrainingDataset ? PlumeLabs88Utils.Load_YTrainPath() : PlumeLabs88Utils.Load_OutputTestRandomPath();
        var y_IDs = df.StringColumnContent(PlumeLabs88Utils.ID_COLUMN_NAME);
        var y_ID_count = NumClass * (1 + DatasetMaxId(isTrainingDataset));
        return y_IDs.Take(y_ID_count).ToArray();
    }
    public override DataFrame Predictions_InModelFormat_2_Predictions_InTargetFormat(DataFrame predictions_InModelFormat, Objective_enum objective)
    {
        AssertNoIdColumns(predictions_InModelFormat);
        var unnormalizedContent = predictions_InModelFormat.FloatTensor.ContentAsFloatArray().Select(UnNormalizeTarget).ToArray();

        return DataFrame.New(unnormalizedContent, TargetLabels);
    }

    public override DataSet TestDataset()
    {
        return new PlumeLabs88DirectoryDataSet(this, false);
    }

    public override DataSet FullTrainingAndValidation()
    {
        return new PlumeLabs88DirectoryDataSet(this, true);
    }


    public override string[] GetColumnNames()
    {
        if (_cacheColumns != null)
        {
            return _cacheColumns;
        }
        _cacheColumns = Array.Empty<string>();
        return _cacheColumns;
    }
    
    public void LoadElementIdIntoSpan(int elementId, Span<float> span, bool isTraining)
    {
        var shortRes = PlumeLabs88Utils.LoadRawElementId(elementId, isTraining);
        var srcShape = PlumeLabs88Utils.Raw_Shape_CHW;
        var targetShape = X_Shape(1).Skip(1).ToArray();
        Debug.Assert(srcShape.Length == 3);
        Debug.Assert(targetShape.Length == 3);
        Debug.Assert(targetShape[0] == srcShape[0]);

        int srcChannelCount = srcShape[1]* srcShape[2];
        int nexIdxInTarget = 0;

        int missingFromTop = (srcShape[1] - targetShape[1]) / 2;
        int missingFromLeft = (srcShape[2] - targetShape[2]) / 2;

        for (int channel = 0; channel < targetShape[0]; ++channel)
        {
            for (int rowInTarget = 0; rowInTarget < targetShape[1]; ++rowInTarget)
            {
                int nextSrcIndex = channel * srcChannelCount;
                nextSrcIndex += (missingFromTop+ rowInTarget) * srcShape[2];
                nextSrcIndex += missingFromLeft;
                for (int colInTarget = 0; colInTarget < targetShape[2]; ++colInTarget)
                {
                    short srcValue = shortRes[nextSrcIndex++];
                    span[nexIdxInTarget++] = NormalizeFeature(srcValue);
                }
            }
        }
    }

    private static float Normalize(float f, float mult, float mean)
    {
        return mult * (MathF.Log(1 + f) - mean);
    }
    private static float UnNormalize(float f, float mult, float mean)
    {
        return (MathF.Exp(f / mult+ mean) - 1);
    }

    public float NormalizeFeature(float f) { return Normalize(f,NormalizeFeatureMult, NormalizeFeatureMean); }
    public float UnNormalizeFeature(float f) { return UnNormalize(f,NormalizeFeatureMult, NormalizeFeatureMean); }
    public float NormalizeTarget(float f) { return Normalize(f, NormalizeTargetMult, NormalizeTargetMean); }
    public float UnNormalizeTarget(float f) { return UnNormalize(f, NormalizeTargetMult, NormalizeTargetMean); }
}