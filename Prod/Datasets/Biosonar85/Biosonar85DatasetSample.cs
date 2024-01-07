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

    private static string xTrainBin_TRANSFORMERS_3D => "X_train_23168_101_64_1024_512_"+f_min+"_"+f_max+"_"+max_db+".bin";
    private static string yTrainBin_TRANSFORMERS_3D => "Y_train_23168_1_64_1024_512_" + f_min + "_" + f_max + "_" + max_db + ".bin";
    private static string xTestBin_TRANSFORMERS_3D => "X_test_950_101_64_1024_512_" + f_min + "_" + f_max + "_" + max_db +".bin";


    //const float x_train_mean = -37.149295f;       //for train dataset only 
    private const float x_train_mean_TRANSFORMERS_3D = -37.076445f;         // for train+test dataset
    //const float x_train_volatility = 14.906728f;  //for train dataset only 
    private const float x_train_volatility_TRANSFORMERS_3D = 14.84446f;     // for train+test dataset
    private const float x_train_mean_PNG_1CHANNEL_V2 = -57.210965f;    // for train dataset
    //public const float x_test_mean_PNG_1CHANNEL_V2 = -54.409614f;     // for test dataset
    private const float x_train_volatility_PNG_1CHANNEL_V2 = 13.373847f;            // for train dataset
    //public const float x_test_volatility_PNG_1CHANNEL_V2 = 10.493092f;             // for test dataset


    //spectrogram with shape (129,401)
    private const string xTrainBin_PNG_1CHANNEL_V2 = "X_train_23168_129_401_256_128_5000_100000_250.bin";
    private const string yTrainBin_PNG_1CHANNEL_V2 = "Y_train_23168_1_None_256_128_5000_100000_250.bin";
    private const string xTestBin_PNG_1CHANNEL_V2 = "X_test_950_129_401_256_128_5000_100000_250.bin";


    //MEL_SPECTROGRAM with shape (64,401)
    private const string xTrainBin_MEL_SPECTROGRAM_64_401 = "X_train_23168_64_401_256_128_1001_127000_250.bin";
    private const string yTrainBin_MEL_SPECTROGRAM_64_401 = "Y_train_23168_1_64_256_128_1001_127000_250.bin";
    private const string xTestBin_MEL_SPECTROGRAM_64_401 = "X_test_950_64_401_256_128_1001_127000_250.bin";
    private const float x_train_mean_MEL_SPECTROGRAM_64_401 = float.NaN;       // for train dataset
    private const float x_train_volatility_MEL_SPECTROGRAM_64_401 = float.NaN;  // for train dataset

    //MEL_SPECTROGRAM with shape (128,401) , frequencies filtered in [1Khz, 127 KHz]
    private const string xTrainBin_MEL_SPECTROGRAM_128_401 = "X_train_23168_128_401_256_128_1001_127000_250.bin";
    //private const string xAugmentedTrainBin_MEL_SPECTROGRAM_128_401 = "augmented501_X_train_23168_128_401_256_128_1001_127000_250.bin";
    private const string xAugmentedTrainBin_MEL_SPECTROGRAM_128_401 = null;
    private const string yTrainBin_MEL_SPECTROGRAM_128_401 = "Y_train_23168_1_128_256_128_1001_127000_250.bin";
    private const string xTestBin_MEL_SPECTROGRAM_128_401 = "X_test_950_128_401_256_128_1001_127000_250.bin";
    private const float x_train_mean_MEL_SPECTROGRAM_128_401 = -65.779617f;       // for train dataset
    private const float x_train_volatility_MEL_SPECTROGRAM_128_401 = 23.741558f;  // for train dataset

    //MEL_SPECTROGRAM_SMALL with shape (128,401) , frequencies filtered in [1Khz, 127 KHz]
    private const string xTrainBin_MEL_SPECTROGRAM_SMALL_128_401 = "X_train_small_256_128_401_256_128_1001_127000_250.bin";
    private const string xAugmentedTrainBin_MEL_SPECTROGRAM_SMALL_128_401 = "augmented2_X_train_small_256_128_401_256_128_1001_127000_250.bin";
    private const string yTrainBin_MEL_SPECTROGRAM_SMALL_128_401 = "Y_train_small_256_1_128_256_128_1001_127000_250.bin";
    private const string xTestBin_MEL_SPECTROGRAM_SMALL_128_401 = "X_test_950_128_401_256_128_1001_127000_250.bin";
    private const float x_train_mean_MEL_SPECTROGRAM_SMALL_128_401 = float.NaN;       // for train dataset
    private const float x_train_volatility_MEL_SPECTROGRAM_SMALL_128_401 = float.NaN;  // for train dataset


    //MEL_SPECTROGRAM with shape (128,401), no filter on frequencies
    //private const string xTrainBin_MEL_SPECTROGRAM_128_401 = "X_train_23168_128_401_256_128_0_None_250.bin";
    //private const string yTrainBin_MEL_SPECTROGRAM_128_401 = "Y_train_23168_1_128_256_128_0_None_250.bin";
    //private const string xTestBin_MEL_SPECTROGRAM_128_401 = "X_test_950_128_401_256_128_0_None_250.bin";
    //private const float x_train_mean_MEL_SPECTROGRAM_128_401 = float.NaN;       // for train dataset
    //private const float x_train_volatility_MEL_SPECTROGRAM_128_401 = float.NaN;  // for train dataset


    //MEL_SPECTROGRAM with shape (256,801)
    private const string xTrainBin_MEL_SPECTROGRAM_256_801 = "X_train_23168_256_801_128_64_1001_127000_250.bin";
    private const string yTrainBin_MEL_SPECTROGRAM_256_801 = "Y_train_23168_1_256_128_64_1001_127000_250.bin";
    //private const string xTrainBin_MEL_SPECTROGRAM_256_801 = "X_train_small_256_256_801_128_64_1001_127000_250.bin";
    //private const string yTrainBin_MEL_SPECTROGRAM_256_801 = "Y_train_small_256_1_256_128_64_1001_127000_250.bin";
    //public static string Y_train_path = Y_train_small_path;

    private const string xTestBin_MEL_SPECTROGRAM_256_801 = "X_test_950_256_801_128_64_1001_127000_250.bin";
    private const float x_train_mean_MEL_SPECTROGRAM_256_801 = -84.293421f;       // for train dataset
    private const float x_train_volatility_MEL_SPECTROGRAM_256_801 = 20.488616f;  // for train dataset


    private static string xTrain_LIBROSA => Path.Join(Biosonar85Utils.DataDirectory, "X_train_librosa.csv");
    private static string xTest_LIBROSA => Path.Join(Biosonar85Utils.DataDirectory, "X_test_librosa.csv");

    public static string Y_test_path = Path.Join(Biosonar85Utils.DataDirectory, "Y_random_Xwjr6aB.csv");
    // ReSharper disable once UnusedMember.Global
    public static string Y_train_small_path = Path.Join(Biosonar85Utils.DataDirectory, "Y_train_small.csv");

    public static string PNG_train_directory = Path.Join(Biosonar85Utils.DataDirectory, "X_train_64_256_128_1000_150000_250");
    public static string PNG_test_directory = Path.Join(Biosonar85Utils.DataDirectory, "X_test_64_256_128_1000_150000_250");


    public static string Y_train_path = Path.Join(Biosonar85Utils.DataDirectory, "Y_train_ofTdMHi.csv");


    //private static string xTrainBinV2 => "X_train_23168_65_401_128_128_1000_150000_250.bin";
    //private static string yTrainBinV2 => "Y_train_23168_1_None_128_128_1000_150000_250.bin";
    //private static string xTestBinV2 => "X_test_950_65_401_128_128_1000_150000_250.bin";


    #region private fields
    //used for InputDataTypeEnum.TRANSFORMERS_3D && InputDataTypeEnum.NETWORK_4D
    private static DataSet trainDataset_TRANSFORMERS_3D;
    private static DataSet testDataset_TRANSFORMERS_3D;

    private static InMemoryDataSet trainDataset_NETWORK_4D;
    private static InMemoryDataSet testDataset_NETWORK_4D;

    private static DataSet trainDataset_PNG_1CHANNEL;
    private static DataSet testDataset_PNG_1CHANNEL;

    private static DataSet trainDataset_PNG_1CHANNEL_V2;
    private static DataSet testDataset_PNG_1CHANNEL_V2;

    private static DataSet trainDataset_MEL_SPECTROGRAM_64_401;
    private static DataSet testDataset_MEL_SPECTROGRAM_64_401;

    private static DataSet trainDataset_MEL_SPECTROGRAM_128_401;
    private static DataSet testDataset_MEL_SPECTROGRAM_128_401;

    private static DataSet trainDataset_MEL_SPECTROGRAM_SMALL_128_401;
    private static DataSet testDataset_MEL_SPECTROGRAM_SMALL_128_401;

    private static DataSet trainDataset_MEL_SPECTROGRAM_256_801;
    private static DataSet testDataset_MEL_SPECTROGRAM_256_801;

    

    private static DataFrame x_training_LIBROSA_FEATURES;
    private static DataFrame y_training_LIBROSA_FEATURES;
    private static DataFrame x_test_LIBROSA_FEATURES;
    private static DataSet trainDataset_LIBROSA_FEATURES;
    private static DataSet testDataset_LIBROSA_FEATURES;

    #endregion


    #region public fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(Biosonar85DatasetSample));
    #endregion

    public enum InputDataTypeEnum { PNG_1CHANNEL_V2, TRANSFORMERS_3D, NETWORK_4D, LIBROSA_FEATURES, MEL_SPECTROGRAM_64_401, MEL_SPECTROGRAM_128_401, MEL_SPECTROGRAM_SMALL_128_401, MEL_SPECTROGRAM_256_801 }

    


    #region Hyperparameters
    public InputDataTypeEnum InputDataType;

    /// <summary>
    /// the site that must be used for training (other sites will be ised for validation)
    /// if empty, we'll not rely on this list for splitting train and validation
    /// </summary>
    public string MandatorySitesForTraining = null;

    #endregion




    public Biosonar85DatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(Biosonar85Utils.WorkingDirectory, Biosonar85Utils.NAME);
        Utils.ConfigureThreadLog4netProperties(Biosonar85Utils.WorkingDirectory, Biosonar85Utils.NAME);
    }

    public override int NumClass => 1;
    public override string IdColumn => "id";
    public override string[] TargetLabels { get; } = { "pos_label" };
    public override bool IsCategoricalColumn(string columnName) => DefaultIsCategoricalColumn(columnName);
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }
    public override DataSet FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }
    public override DataSet TestDataset()
    {
        return LoadAndEncodeDataset_If_Needed().testDataset;
    }

    private ITrainingAndTestDataset SplitIntoTrainingAndValidationFromMandatorySitesForTraining()
    {
        Debug.Assert(!string.IsNullOrEmpty(MandatorySitesForTraining));
        var sitesForTraining = new HashSet<string>(MandatorySitesForTraining.Split(","));
        var fullTrain = FullTrainingAndValidation();
        List<int> idInTrainingList = new();
        for (int i = 0; i < fullTrain.Y_IDs.Length; ++i)
        {
            var id = fullTrain.Y_IDs[i];
            var site = Biosonar85Utils.IdToSite(id);
            if (sitesForTraining.Contains(site))
            {
                idInTrainingList.Add(i);
            }
        }
        Log.Info($"Using following {idInTrainingList.Count} sites for training {MandatorySitesForTraining}");

        if (ShuffleDatasetBeforeSplit)
        {
            Utils.Shuffle(idInTrainingList, fullTrain.FirstRandom);
        }
        var idInTrainingSet = new HashSet<int>(idInTrainingList);
        var training = fullTrain.SubDataSet(id => idInTrainingSet.Contains(id));
        var validation = fullTrain.SubDataSet(id => !idInTrainingSet.Contains(id));
        return new TrainingAndTestDataset(training, validation, Name);
    }



    public override ITrainingAndTestDataset SplitIntoTrainingAndValidation()
    {
        if (!string.IsNullOrEmpty(MandatorySitesForTraining))
        {
            return SplitIntoTrainingAndValidationFromMandatorySitesForTraining();
        }

        var fullTrain = FullTrainingAndValidation();
        int rowsForTraining = (int)(PercentageInTraining * fullTrain.Count + 0.1);

        var siteToY_Id_indexes = new Dictionary<string, List<int>>();
        for (int i = 0; i < fullTrain.Y_IDs.Length; ++i)
        {
            var id = fullTrain.Y_IDs[i];
            var site = Biosonar85Utils.IdToSite(id);
            if (!siteToY_Id_indexes.ContainsKey(site))
            {
                siteToY_Id_indexes[site] = new List<int>();
            }
            siteToY_Id_indexes[site].Add(i);
        }

        KeyValuePair<string, List<int>>[] sortedSites = siteToY_Id_indexes.OrderByDescending(v => v.Value.Count).ToArray();
        List<int> idInTrainingList = new();
        List<string> sitesInTrainingList = new();
        idInTrainingList.AddRange(sortedSites[0].Value);
        sitesInTrainingList.Add(sortedSites[0].Key);
        for (int siteIndex = 1; siteIndex < sortedSites.Length; ++siteIndex)
        {
            var newSiteCount = sortedSites[siteIndex].Value.Count;
            int errorWithoutNewSite = Math.Abs(idInTrainingList.Count - rowsForTraining);
            int errorWitNewSite = Math.Abs(idInTrainingList.Count + newSiteCount - rowsForTraining);
            if (errorWithoutNewSite <= errorWitNewSite)
            {
                Log.Info($"Using following {idInTrainingList.Count} sites for training {string.Join(',', sitesInTrainingList)}");
                break;
            }
            idInTrainingList.AddRange(sortedSites[siteIndex].Value);
            sitesInTrainingList.Add(sortedSites[siteIndex].Key);
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

    public override int[] X_Shape(int batchSize)
    {
        switch (InputDataType)
        {
            case InputDataTypeEnum.TRANSFORMERS_3D:
                return X_Shape(xTrainBin_TRANSFORMERS_3D, batchSize).ToArray();
            case InputDataTypeEnum.NETWORK_4D:
                return X_Shape(xTrainBin_TRANSFORMERS_3D, batchSize);
            case InputDataTypeEnum.PNG_1CHANNEL_V2:
                return X_Shape(xTrainBin_PNG_1CHANNEL_V2, batchSize);
            case InputDataTypeEnum.MEL_SPECTROGRAM_64_401:
                return X_Shape(xTrainBin_MEL_SPECTROGRAM_64_401, batchSize);
            case InputDataTypeEnum.MEL_SPECTROGRAM_128_401:
                return X_Shape(xTrainBin_MEL_SPECTROGRAM_128_401, batchSize);
            case InputDataTypeEnum.MEL_SPECTROGRAM_SMALL_128_401:
                return X_Shape(xTrainBin_MEL_SPECTROGRAM_SMALL_128_401, batchSize);
            case InputDataTypeEnum.MEL_SPECTROGRAM_256_801:
                return X_Shape(xTrainBin_MEL_SPECTROGRAM_256_801, batchSize);
            case InputDataTypeEnum.LIBROSA_FEATURES:
                throw new ArgumentException();
            default:
                throw new NotImplementedException($"{InputDataType}");
        }
    }

    private static int[] X_Shape(string path, int batchSize)
    {
        var x_shape = (int[])Biosonar85Utils.ProcessXFileName(path).shape.Clone();
        x_shape[0] = batchSize;
        return x_shape;
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
                        trainDataset_TRANSFORMERS_3D = Biosonar85Utils.Load(xTrainBin_TRANSFORMERS_3D, yTrainBin_TRANSFORMERS_3D, Y_train_path, x_train_mean_TRANSFORMERS_3D, x_train_volatility_TRANSFORMERS_3D);
                        testDataset_TRANSFORMERS_3D = Biosonar85Utils.Load(xTestBin_TRANSFORMERS_3D, null, Y_test_path, x_train_mean_TRANSFORMERS_3D, x_train_volatility_TRANSFORMERS_3D);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_TRANSFORMERS_3D, testDataset_TRANSFORMERS_3D);
                case InputDataTypeEnum.NETWORK_4D:
                    if (trainDataset_NETWORK_4D == null)
                    {
                        trainDataset_NETWORK_4D = Biosonar85Utils.Load(xTrainBin_TRANSFORMERS_3D, yTrainBin_TRANSFORMERS_3D, Y_train_path, x_train_mean_TRANSFORMERS_3D, x_train_volatility_TRANSFORMERS_3D);
                        testDataset_NETWORK_4D = Biosonar85Utils.Load(xTestBin_TRANSFORMERS_3D, null, Y_test_path, x_train_mean_TRANSFORMERS_3D, x_train_volatility_TRANSFORMERS_3D);
                        trainDataset_NETWORK_4D.X.ReshapeInPlace(trainDataset_NETWORK_4D.X.Shape[0], 1, trainDataset_NETWORK_4D.X.Shape[2], trainDataset_NETWORK_4D.X.Shape[3]);
                        testDataset_NETWORK_4D.X.ReshapeInPlace(testDataset_NETWORK_4D.X.Shape[0], 1, testDataset_NETWORK_4D.X.Shape[2], testDataset_NETWORK_4D.X.Shape[3]);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_NETWORK_4D, testDataset_NETWORK_4D);
                case InputDataTypeEnum.PNG_1CHANNEL_V2:
                    if (trainDataset_PNG_1CHANNEL_V2 == null)
                    {
                        //trainDataset_PNG_1CHANNEL_V2 = Biosonar85Utils.Load("X_train_small_10_129_401_256_128_1000_150000_250.bin", "Y_train_small_10_1_None_256_128_1000_150000_250.bin", Y_train_small_path, Biosonar85Utils.x_train_mean_PNG_1CHANNEL_V2, Biosonar85Utils.x_train_volatility_PNG_1CHANNEL_V2);
                        trainDataset_PNG_1CHANNEL_V2 = Biosonar85Utils.LoadTensorListDataSet(xTrainBin_PNG_1CHANNEL_V2, null, yTrainBin_PNG_1CHANNEL_V2, Y_train_path, x_train_mean_PNG_1CHANNEL_V2, x_train_volatility_PNG_1CHANNEL_V2);
                        testDataset_PNG_1CHANNEL_V2 = Biosonar85Utils.LoadTensorListDataSet(xTestBin_PNG_1CHANNEL_V2, null, null, Y_test_path, x_train_mean_PNG_1CHANNEL_V2, x_train_volatility_PNG_1CHANNEL_V2);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_PNG_1CHANNEL_V2, testDataset_PNG_1CHANNEL_V2);

                case InputDataTypeEnum.MEL_SPECTROGRAM_64_401:
                    if (trainDataset_MEL_SPECTROGRAM_64_401 == null)
                    {
                        trainDataset_MEL_SPECTROGRAM_64_401 = Biosonar85Utils.LoadTensorListDataSet(xTrainBin_MEL_SPECTROGRAM_64_401, null, yTrainBin_MEL_SPECTROGRAM_64_401, Y_train_path, x_train_mean_MEL_SPECTROGRAM_64_401, x_train_volatility_MEL_SPECTROGRAM_64_401);
                        testDataset_MEL_SPECTROGRAM_64_401 = Biosonar85Utils.LoadTensorListDataSet(xTestBin_MEL_SPECTROGRAM_64_401, null, null, Y_test_path, x_train_mean_MEL_SPECTROGRAM_64_401, x_train_volatility_MEL_SPECTROGRAM_64_401);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_MEL_SPECTROGRAM_64_401, testDataset_MEL_SPECTROGRAM_64_401);

                case InputDataTypeEnum.MEL_SPECTROGRAM_128_401:
                    if (trainDataset_MEL_SPECTROGRAM_128_401 == null)
                    {
                        trainDataset_MEL_SPECTROGRAM_128_401 = Biosonar85Utils.LoadTensorListDataSet(xTrainBin_MEL_SPECTROGRAM_128_401, xAugmentedTrainBin_MEL_SPECTROGRAM_128_401, yTrainBin_MEL_SPECTROGRAM_128_401, Y_train_path, x_train_mean_MEL_SPECTROGRAM_128_401, x_train_volatility_MEL_SPECTROGRAM_128_401);
                        testDataset_MEL_SPECTROGRAM_128_401 = Biosonar85Utils.LoadTensorListDataSet(xTestBin_MEL_SPECTROGRAM_128_401, null, null, Y_test_path, x_train_mean_MEL_SPECTROGRAM_128_401, x_train_volatility_MEL_SPECTROGRAM_128_401);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_MEL_SPECTROGRAM_128_401, testDataset_MEL_SPECTROGRAM_128_401);

                case InputDataTypeEnum.MEL_SPECTROGRAM_SMALL_128_401:
                    if (trainDataset_MEL_SPECTROGRAM_SMALL_128_401 == null)
                    {
                        trainDataset_MEL_SPECTROGRAM_SMALL_128_401 = Biosonar85Utils.LoadTensorListDataSet(xTrainBin_MEL_SPECTROGRAM_SMALL_128_401, xAugmentedTrainBin_MEL_SPECTROGRAM_SMALL_128_401, yTrainBin_MEL_SPECTROGRAM_SMALL_128_401, Y_train_small_path, x_train_mean_MEL_SPECTROGRAM_SMALL_128_401, x_train_volatility_MEL_SPECTROGRAM_SMALL_128_401);
                        testDataset_MEL_SPECTROGRAM_SMALL_128_401 = Biosonar85Utils.LoadTensorListDataSet(xTestBin_MEL_SPECTROGRAM_SMALL_128_401, null, null, Y_test_path, x_train_mean_MEL_SPECTROGRAM_SMALL_128_401, x_train_volatility_MEL_SPECTROGRAM_SMALL_128_401);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_MEL_SPECTROGRAM_SMALL_128_401, testDataset_MEL_SPECTROGRAM_SMALL_128_401);

                case InputDataTypeEnum.MEL_SPECTROGRAM_256_801:
                    if (trainDataset_MEL_SPECTROGRAM_256_801 == null)
                    {
                        trainDataset_MEL_SPECTROGRAM_256_801 = Biosonar85Utils.LoadTensorListDataSet(xTrainBin_MEL_SPECTROGRAM_256_801, null, yTrainBin_MEL_SPECTROGRAM_256_801, Y_train_path, x_train_mean_MEL_SPECTROGRAM_256_801, x_train_volatility_MEL_SPECTROGRAM_256_801);
                        testDataset_MEL_SPECTROGRAM_256_801 = Biosonar85Utils.LoadTensorListDataSet(xTestBin_MEL_SPECTROGRAM_256_801, null, null, Y_test_path, x_train_mean_MEL_SPECTROGRAM_256_801, x_train_volatility_MEL_SPECTROGRAM_256_801);
                        Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
                    }
                    return (trainDataset_MEL_SPECTROGRAM_256_801, testDataset_MEL_SPECTROGRAM_256_801);
                    

                case InputDataTypeEnum.LIBROSA_FEATURES:
                    if (trainDataset_LIBROSA_FEATURES == null)
                    {
                        x_training_LIBROSA_FEATURES = DataFrame.read_csv_normalized(xTrain_LIBROSA, ',', true, ColumnNameToType);
                        y_training_LIBROSA_FEATURES = DataFrame.read_csv_normalized(Y_train_path, ',', true, ColumnNameToType);
                        trainDataset_LIBROSA_FEATURES = new DataFrameDataSet(this,
                            x_training_LIBROSA_FEATURES.Drop("id").Clone(),
                            y_training_LIBROSA_FEATURES.Drop("id").Clone(),
                            x_training_LIBROSA_FEATURES.StringColumnContent("id"));
                        x_test_LIBROSA_FEATURES = DataFrame.read_csv_normalized(xTest_LIBROSA, ',', true, ColumnNameToType);
                        testDataset_LIBROSA_FEATURES = new DataFrameDataSet(this,
                            x_test_LIBROSA_FEATURES.Drop("id").Clone(),
                            null,
                            x_training_LIBROSA_FEATURES.StringColumnContent("id"));
                    }
                    return (trainDataset_LIBROSA_FEATURES, testDataset_LIBROSA_FEATURES);
                default:
                    throw new NotImplementedException($"{InputDataType}");
            }
        }
    }

    public static Type ColumnNameToType(string columnName)
    {
        if (columnName.Equals("id"))
        {
            return typeof(string);
        }
        return typeof(float);
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