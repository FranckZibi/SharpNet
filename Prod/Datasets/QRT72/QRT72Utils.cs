using System.IO;

namespace SharpNet.Datasets.QRT72;

public static class QRT72Utils
{
    public const string NAME = "QRT72";
    public const int D = 250; //250 days
    public const int F = 10; //10 features

    #region load of datasets
    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    // ReSharper disable once MemberCanBePrivate.Global
    #endregion

}