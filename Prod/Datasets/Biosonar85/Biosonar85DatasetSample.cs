using System;
using log4net;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.MathTools;
// ReSharper disable FieldCanBeMadeReadOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.Biosonar85;

public class Biosonar85DatasetSample : AbstractDatasetSample
{
    private const string xTrainBin = "X_train_23168_101_64_1024_512.bin";
    private const string xTestBin = "X_test_950_101_64_1024_512.bin";
    private const string yTrainBin = "Y_train_23168_1_64_1024_512.bin";

    #region private fields
    private static readonly InMemoryDataSet trainDataset;
    private static readonly InMemoryDataSet testDataset;
    #endregion
    
    #region public fields & properties

    private static readonly ILog Log = LogManager.GetLogger(typeof(Biosonar85DatasetSample));
    #endregion

    #region HyperParameters
    public bool UseTransformers = false;
    #endregion


    static Biosonar85DatasetSample()
    {
        var sw = Stopwatch.StartNew();
        Log.Debug($"Starting loading raw files");
        trainDataset = Biosonar85Utils.Load(xTrainBin, yTrainBin, "Y_train_ofTdMHi.csv");
        //trainDataset = Biosonar85Utils.Load("X_train_small_1000_101_64_1024_512.bin", "Y_train_small_1000_1_64_1024_512.bin", "Y_train_small.csv", true);
        //AddToDispose(trainDataset);
        testDataset = Biosonar85Utils.Load(xTestBin, null, "Y_random_Xwjr6aB.csv");
        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
    }

    public Biosonar85DatasetSample() : base(new HashSet<string>())
    {
        Utils.ConfigureGlobalLog4netProperties(Biosonar85Utils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(Biosonar85Utils.WorkingDirectory, "log");
    }

    public override string[] CategoricalFeatures { get; } = { };
    public override string IdColumn => "id";
    public override string[] TargetLabels { get; } = { "pos_label" };
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }
    //public override IScore MinimumScoreToSaveModel => new Score(0.92f, GetRankingEvaluationMetric());

    public override int NumClass => 1;
    public override string[] TargetLabelDistinctValues => Biosonar85Utils.TargetLabelDistinctValues;
    
    public override InMemoryDataSet FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }
    public override InMemoryDataSet TestDataset()
    {
        return LoadAndEncodeDataset_If_Needed().testDataset;
    }

    public override ITrainingAndTestDataset SplitIntoTrainingAndValidation()
    {
        var fullTrain = FullTrainingAndValidation();
        int rowsForTraining = (int)(PercentageInTraining * fullTrain.Count + 0.1);

        string IdToSite(string id) { return id.Split(new[] { '-', '.' })[1]; }
        var siteToY_Id_indexes = new Dictionary<string, List<int>>();
        for (int i = 0; i < fullTrain.Y_IDs.Length; ++i)
        {
            var id = fullTrain.Y_IDs[i];
            var site = IdToSite(id);
            if (!siteToY_Id_indexes.ContainsKey(site))
            {
                siteToY_Id_indexes[site] = new List<int>();
            }
            siteToY_Id_indexes[site].Add(i);
        }

        List<int>[] sortedSites = siteToY_Id_indexes.OrderByDescending(v => v.Value.Count).Select(t=>t.Value).ToArray();
        List<int> idInTrainingList = new();
        idInTrainingList.AddRange(sortedSites[0]);
        var currentCountInTrain = idInTrainingList.Count;
        for (int siteIndex = 1; siteIndex < (sortedSites.Length-1); ++siteIndex)
        {
            var newSiteCount = sortedSites[siteIndex].Count;
            int errorWithoutNewSite = Math.Abs(currentCountInTrain-rowsForTraining);
            int errorWitNewSite = Math.Abs(currentCountInTrain+ newSiteCount - rowsForTraining);
            if (errorWithoutNewSite <= errorWitNewSite)
            {
                break;
            }
            idInTrainingList.AddRange(sortedSites[siteIndex]);
            currentCountInTrain+= newSiteCount;
        }

        if (ShuffleDatasetBeforeSplit)
        {
            Utils.Shuffle(idInTrainingList, fullTrain.FirstRandom);
        }
        else
        {
            idInTrainingList.Sort();
        }
        var idInTrainingSet = new HashSet<int>(idInTrainingList);
        var training = fullTrain.SubDataSet(id => idInTrainingSet.Contains(id));
        var validation = fullTrain.SubDataSet(id => !idInTrainingSet.Contains(id));
        return new TrainingAndTestDataset(training, validation, Name);
    }


    private (InMemoryDataSet fullTrainingAndValidation, InMemoryDataSet testDataset) LoadAndEncodeDataset_If_Needed()
    {
        if (UseTransformers && trainDataset.X.Shape.Length == 4)
        {
            trainDataset.X.ReshapeInPlace(trainDataset.X.Shape[0], trainDataset.X.Shape[2], trainDataset.X.Shape[3]);
            testDataset.X.ReshapeInPlace(testDataset.X.Shape[0], testDataset.X.Shape[2], testDataset.X.Shape[3]);
        }
        if (!UseTransformers && trainDataset.X.Shape.Length == 3)
        {
            trainDataset.X.ReshapeInPlace(trainDataset.X.Shape[0], 1, trainDataset.X.Shape[1], trainDataset.X.Shape[2]);
            testDataset.X.ReshapeInPlace(testDataset.X.Shape[0], 1, testDataset.X.Shape[1], testDataset.X.Shape[2]);
        }

        return (trainDataset, testDataset);
    }

    /// <summary>
    /// compute stats for train & test dataset
    /// </summary>
    // ReSharper disable once UnusedMember.Local
    private static void ComputeStats()
    {
        var xTrainPath = Path.Join(Biosonar85Utils.DataDirectory, xTrainBin);
        (int[] xTrainShape, var _, var _) = Biosonar85Utils.ProcessXFileName(xTrainPath);
        var xTrainTensor = CpuTensor<float>.LoadFromBinFile(xTrainPath, xTrainShape);
        var xTrainAcc = new DoubleAccumulator();
        xTrainAcc.Add(xTrainTensor.ContentAsFloatArray());
        Log.Info($"Stats for {xTrainPath} before standardization: {xTrainAcc}");

        var xTestPath = Path.Join(Biosonar85Utils.DataDirectory, xTestBin);
        (int[] xTestShape, var _, var _) = Biosonar85Utils.ProcessXFileName(xTestPath);
        var xTestTensor = CpuTensor<float>.LoadFromBinFile(xTestPath, xTestShape);
        var xTestAcc = new DoubleAccumulator();
        xTestAcc.Add(xTestTensor.ContentAsFloatArray());
        Log.Info($"Stats for {xTestPath} before standardization: {xTestAcc}");

        Log.Info($"Cumulative Stats Stats for : {DoubleAccumulator.Sum(xTrainAcc, xTestAcc)}");
    }

}