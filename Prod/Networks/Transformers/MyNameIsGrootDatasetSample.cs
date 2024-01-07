using System.Text;
using log4net;
using SharpNet.Datasets;
using SharpNet.TextPreprocessing;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks.Transformers;

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

    private const string sentence = "my name is groot";

    public MyNameIsGrootDatasetSample()
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

    public override DataFrame Predictions_InModelFormat_2_Predictions_InTargetFormat(DataFrame predictions_InModelFormat, Objective_enum objective)
    {
        return predictions_InModelFormat;
    }

    public override string IdColumn => null;
    public override string[] TargetLabels { get; } = {"y"};
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
            MyNameIsGrootUtils.NAME,
            _fullText,
            GetTokenizer());
    }

    public string GetText() => _fullText;
}