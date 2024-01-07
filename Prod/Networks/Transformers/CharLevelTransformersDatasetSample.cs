﻿using System.IO;
using log4net;
using SharpNet.Datasets;
using SharpNet.TextPreprocessing;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks.Transformers;

public class CharLevelTransformersDatasetSample : TransformerDatasetSample
{
    #region private static fields

    private readonly string _fullText;

    #endregion

    #region public fields & properties
    // ReSharper disable once UnusedMember.Global
    public static readonly ILog Log = LogManager.GetLogger(typeof(CharLevelTransformersDatasetSample));
    #endregion

    #region Hyperparameters
    /// <summary>
    /// the maximum size fo the text that will be used for training the network
    /// -1 means the full text (default value)
    /// other values are used to speed up the training process (usually for testing purpose)
    /// </summary>
    // ReSharper disable once ConvertToConstant.Global
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public int MaxCharacterLengthForTraining = -1;

    #endregion

    static CharLevelTransformersDatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(TextTransformersUtils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(TextTransformersUtils.WorkingDirectory, "log");
    }

    public CharLevelTransformersDatasetSample()
    {
        //_fullText = File.ReadAllText(TextTransformersUtils.XTrainPath).Substring(0, 3_000); //!D;
        _fullText = File.ReadAllText(TextTransformersUtils.XTrainPath);
    }

    public Tokenizer GetTokenizer()
    {
        var tokenizer = new Tokenizer(vocab_size, oovToken: null, char_level: true, lowerCase: false, filters: "\r",
            firstElementIsPaddingToken: false);
        tokenizer.FitOnTexts(new[] { _fullText });
        return tokenizer;
    }

    public string GetText(int maxCharacters = -1)
    {
        if (maxCharacters != -1 && maxCharacters < _fullText.Length)
        {
            return _fullText.Substring(0, maxCharacters);
        }
        return _fullText;
    }

    public override DataFrame Predictions_InModelFormat_2_Predictions_InTargetFormat(DataFrame predictions_InModelFormat, Objective_enum objective)
    {
        return predictions_InModelFormat;
    }
    public override string IdColumn => null;
    public override string[] TargetLabels { get; } = null;
    public override bool IsCategoricalColumn(string columnName) => DefaultIsCategoricalColumn(columnName);
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }
    public override DataSet TestDataset()
    {
        return null;
    }

    public override DataSet FullTrainingAndValidation()
    {
        return new TransformerDataset(
            this,
            "CharLevel",
            GetText(MaxCharacterLengthForTraining),
            GetTokenizer());
    }
}