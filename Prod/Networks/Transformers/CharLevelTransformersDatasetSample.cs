using System.Collections.Generic;
using System.IO;
using log4net;
using SharpNet.Datasets;
using SharpNet.TextPreprocessing;

namespace SharpNet.Networks.Transformers;

public class CharLevelTransformersDatasetSample : AbstractDatasetSample
{
    #region private static fields

    private readonly string _fullText;

    #endregion

    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(CharLevelTransformersDatasetSample));
    #endregion

    #region HyperParameters
    public int vocab_size = 65;
    public int max_length = 32; // == timeSteps

    /// <summary>
    /// the maximum size fo the text that will be used for training the network
    /// -1 means the full text (default value)
    /// other values are used to speed up the training process (usually for testing purpose)
    /// </summary>
    public int MaxCharacterLengthForTraining = -1;

    #endregion

    static CharLevelTransformersDatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(TextTransformersUtils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(TextTransformersUtils.WorkingDirectory, "log");
    }

    public CharLevelTransformersDatasetSample() : base(new HashSet<string>())
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

    public override int NumClass => vocab_size;
    public override int[] GetInputShapeOfSingleElement() => new[] { max_length };
    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat, Objective_enum objective)
    {
        return predictionsInModelFormat;
    }

    public override string[] CategoricalFeatures { get; } = { "" };
    public override string IdColumn => null;
    public override string[] TargetLabels { get; } = null;
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }

    public override string[] TargetLabelDistinctValues => null;

    public override DataSet TestDataset()
    {
        return null;
    }

    public override DataSet FullTrainingAndValidation()
    {
        return new CharLevelDataset(
            this,
            "CharLevel",
            GetText(MaxCharacterLengthForTraining),
            GetTokenizer());
    }
}