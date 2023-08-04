using System;
using log4net;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
// ReSharper disable FieldCanBeMadeReadOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.Biosonar85;

public class Biosonar85DatasetSample : AbstractDatasetSample
{
    public const string f_min = "1000";
    public const string f_max = "150000";
    public const string max_db = "250";

    private static string xTrainBin => "X_train_23168_101_64_1024_512_"+f_min+"_"+f_max+"_"+max_db+".bin";
    private static string yTrainBin => "Y_train_23168_1_64_1024_512_" + f_min + "_" + f_max + "_" + max_db + ".bin";
    private static string xTestBin => "X_test_950_101_64_1024_512_" + f_min + "_" + f_max + "_" + max_db +".bin";

    private static string xTrainBinV2 => "X_train_23168_129_401_256_128_1000_150000_250.bin";
    private static string yTrainBinV2 => "Y_train_23168_1_64_256_128_1000_150000_250.bin";
    private static string xTestBinV2 => "X_test_950_129_401_256_128_1000_150000_250.bin";

    #region private fields
    //used for InputDataTypeEnum.TRANSFORMERS_3D && InputDataTypeEnum.NETWORK_4D
    private static InMemoryDataSet trainDataset_TRANSFORMERS_3D;
    private static InMemoryDataSet testDataset_TRANSFORMERS_3D;

    private static InMemoryDataSet trainDataset_NETWORK_4D;
    private static InMemoryDataSet testDataset_NETWORK_4D;

    private static DirectoryDataSet trainDataset_PNG_1CHANNEL;
    private static DirectoryDataSet testDataset_PNG_1CHANNEL;

    private static InMemoryDataSet trainDataset_PNG_1CHANNEL_V2;
    private static InMemoryDataSet testDataset_PNG_1CHANNEL_V2;
    #endregion


    #region public fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(Biosonar85DatasetSample));
    #endregion

    public enum InputDataTypeEnum { PNG_1CHANNEL, PNG_1CHANNEL_V2, TRANSFORMERS_3D, NETWORK_4D}

    #region HyperParameters
    public InputDataTypeEnum InputDataType;
    #endregion


    public static string Y_train_path = Path.Join(Biosonar85Utils.DataDirectory, "Y_train_ofTdMHi.csv");
    public static string Y_test_path = Path.Join(Biosonar85Utils.DataDirectory, "Y_random_Xwjr6aB.csv");
    // ReSharper disable once UnusedMember.Global
    public static string Y_train_small_path = Path.Join(Biosonar85Utils.DataDirectory, "Y_train_small.csv");

    public static string PNG_train_directory = Path.Join(Biosonar85Utils.DataDirectory, "X_train_64_256_128_1000_150000_250");
    public static string PNG_test_directory = Path.Join(Biosonar85Utils.DataDirectory, "X_test_64_256_128_1000_150000_250");


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
    public override int NumClass => 1;
    public override string[] TargetLabelDistinctValues => Biosonar85Utils.TargetLabelDistinctValues;
    
    public override DataSet FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }
    public override DataSet TestDataset()
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
        for (int siteIndex = 1; siteIndex < sortedSites.Length; ++siteIndex)
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

    public override int[] GetInputShapeOfSingleElement()
    {
        switch (InputDataType)
        {
            case InputDataTypeEnum.TRANSFORMERS_3D:
                return new[] {101, 64 };
            case InputDataTypeEnum.NETWORK_4D:
                return new[] {1, 101, 64 };
            case InputDataTypeEnum.PNG_1CHANNEL:
                return new[] { 1, 129, 401 };
            case InputDataTypeEnum.PNG_1CHANNEL_V2:
                return new[] { 1, 129, 401 };
            default:
                throw new NotImplementedException($"{InputDataType}");
        }
    }

    private readonly object _lockObject = new ();
    private (DataSet fullTrainingAndValidation, DataSet testDataset) LoadAndEncodeDataset_If_Needed()
    {
        lock (_lockObject)
        {
            var sw = Stopwatch.StartNew();
            switch (InputDataType)
            {
                case InputDataTypeEnum.TRANSFORMERS_3D:
                    if (trainDataset_TRANSFORMERS_3D == null)
                    {
                        

                        trainDataset_TRANSFORMERS_3D = Biosonar85Utils.Load(xTrainBin, yTrainBin, Y_train_path, Biosonar85Utils.x_train_mean, Biosonar85Utils.x_train_volatility);
                        testDataset_TRANSFORMERS_3D = Biosonar85Utils.Load(xTestBin, null, Y_test_path, Biosonar85Utils.x_train_mean, Biosonar85Utils.x_train_volatility);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_TRANSFORMERS_3D, testDataset_TRANSFORMERS_3D);
                case InputDataTypeEnum.NETWORK_4D:
                    if (trainDataset_NETWORK_4D == null)
                    {
                        trainDataset_NETWORK_4D = Biosonar85Utils.Load(xTrainBin, yTrainBin, Y_train_path, Biosonar85Utils.x_train_mean, Biosonar85Utils.x_train_volatility);
                        testDataset_NETWORK_4D = Biosonar85Utils.Load(xTestBin, null, Y_test_path, Biosonar85Utils.x_train_mean, Biosonar85Utils.x_train_volatility);
                        trainDataset_NETWORK_4D.X.ReshapeInPlace(trainDataset_NETWORK_4D.X.Shape[0], 1, trainDataset_NETWORK_4D.X.Shape[2], trainDataset_NETWORK_4D.X.Shape[3]);
                        testDataset_NETWORK_4D.X.ReshapeInPlace(testDataset_NETWORK_4D.X.Shape[0], 1, testDataset_NETWORK_4D.X.Shape[2], testDataset_NETWORK_4D.X.Shape[3]);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_NETWORK_4D, testDataset_NETWORK_4D);
                case InputDataTypeEnum.PNG_1CHANNEL:
                    if (trainDataset_PNG_1CHANNEL == null)
                    {
                        var meanAndVolatilityForEachChannelTrain = new List<Tuple<float, float>> { Tuple.Create(121.41582f, 38.465096f) };
                        trainDataset_PNG_1CHANNEL = Biosonar85Utils.LoadPng(PNG_train_directory, Y_train_path, true, meanAndVolatilityForEachChannelTrain);
                        var meanAndVolatilityForEachChannelTest = new List<Tuple<float, float>> { Tuple.Create(115.38992f, 27.096777f) };
                        meanAndVolatilityForEachChannelTest = meanAndVolatilityForEachChannelTrain;
                        testDataset_PNG_1CHANNEL = Biosonar85Utils.LoadPng(PNG_test_directory, Y_test_path, false, meanAndVolatilityForEachChannelTest);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_PNG_1CHANNEL, testDataset_PNG_1CHANNEL);
                case InputDataTypeEnum.PNG_1CHANNEL_V2:
                    if (trainDataset_PNG_1CHANNEL_V2 == null)
                    {
                        //trainDataset_PNG_1CHANNEL_V2 = Biosonar85Utils.Load("X_train_small_10_129_401_256_128_1000_150000_250.bin", "Y_train_small_10_1_None_256_128_1000_150000_250.bin", Y_train_small_path, Biosonar85Utils.x_train_mean_PNG_1CHANNEL_V2, Biosonar85Utils.x_train_volatility_PNG_1CHANNEL_V2);
                        trainDataset_PNG_1CHANNEL_V2 = Biosonar85Utils.Load(xTrainBinV2, yTrainBinV2, Y_train_path, Biosonar85Utils.x_train_mean_PNG_1CHANNEL_V2, Biosonar85Utils.x_train_volatility_PNG_1CHANNEL_V2);

                        testDataset_PNG_1CHANNEL_V2 = Biosonar85Utils.Load(xTestBinV2, null, Y_test_path, Biosonar85Utils.x_train_mean_PNG_1CHANNEL_V2, Biosonar85Utils.x_train_volatility_PNG_1CHANNEL_V2);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_PNG_1CHANNEL_V2, testDataset_PNG_1CHANNEL_V2);
                default:
                    throw new NotImplementedException($"{InputDataType}");
            }
        }
    }

    ///// <summary>
    ///// compute stats for train & test dataset
    ///// </summary>
    //// ReSharper disable once UnusedMember.Local
    //private static void ComputeStats()
    //{
    //    var xTrainPath = Path.Join(Biosonar85Utils.DataDirectory, xTrainBin);
    //    (int[] xTrainShape, var _, var _, var _, var _, var _) = Biosonar85Utils.ProcessXFileName(xTrainPath);
    //    var xTrainTensor = CpuTensor<float>.LoadFromBinFile(xTrainPath, xTrainShape);
    //    var xTrainAcc = new DoubleAccumulator();
    //    xTrainAcc.Add(xTrainTensor.SpanContent);
    //    Log.Info($"Stats for {xTrainPath} before standardization: {xTrainAcc}");

    //    var xTestPath = Path.Join(Biosonar85Utils.DataDirectory, xTestBin);
    //    (int[] xTestShape, var _, var _, var _, var _, var _) = Biosonar85Utils.ProcessXFileName(xTestPath);
    //    var xTestTensor = CpuTensor<float>.LoadFromBinFile(xTestPath, xTestShape);
    //    var xTestAcc = new DoubleAccumulator();
    //    xTestAcc.Add(xTestTensor.SpanContent);
    //    Log.Info($"Stats for {xTestPath} before standardization: {xTestAcc}");

    //    Log.Info($"Cumulative Stats Stats for : {DoubleAccumulator.Sum(xTrainAcc, xTestAcc)}");
    //}

}