using System.Collections.Generic;
using System.Text;
using log4net;
using SharpNet.Datasets;
using SharpNet.TextPreprocessing;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks.Transformers;

public abstract class TransformerDatasetSample : AbstractDatasetSample
{
    #region HyperParameters
    public int vocab_size = 4;
    public int max_length = 3; // == timeSteps
    #endregion


    protected TransformerDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
    }


    public override int[] GetInputShapeOfSingleElement() => new[] { max_length };
    public override int[] X_Shape(int batchSize) => new[] { batchSize, max_length };
    public override int[] Y_Shape(int batchSize) => new[] { batchSize, vocab_size };
}

public class MyNameIsGrootDatasetSample : TransformerDatasetSample
{
    #region private static fields

    private readonly string _fullText;

    #endregion

    #region public fields & properties
    // ReSharper disable once UnusedMember.Global
    public static readonly ILog Log = LogManager.GetLogger(typeof(MyNameIsGrootDatasetSample));
    #endregion


    static MyNameIsGrootDatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(MyNameIsGrootUtils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(MyNameIsGrootUtils.WorkingDirectory, "log");
    }

    private readonly string sentence = "my name is groot";

    public MyNameIsGrootDatasetSample() : base(new HashSet<string>())
    {
        var sb = new StringBuilder();
        for (int i = 0; i < 20_000; ++i)
        //for (int i = 0; i < 2; ++i)
        {
            sb.Append(sentence + " ");
        }
        _fullText = sb.ToString();
    }

    public Tokenizer GetTokenizer()
    {
        var tokenizer = new Tokenizer(vocab_size, oovToken: null, char_level: false, lowerCase: true, firstElementIsPaddingToken: false);
        tokenizer.FitOnTexts(new[] { _fullText });
        return tokenizer;
    }
    
    public override int NumClass => vocab_size;

    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat, Objective_enum objective)
    {
        return predictionsInModelFormat;
    }

    public override string[] CategoricalFeatures { get; } = { "" };
    public override string IdColumn => null;
    public override string[] TargetLabels { get; } = {"y"};
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
        return new TransformerDataset(
            this,
            MyNameIsGrootUtils.NAME,
            _fullText,
            GetTokenizer());
    }

    public string GetText() => _fullText;
}