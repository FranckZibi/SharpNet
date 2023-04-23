using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;
using SharpNet.CPU;
using SharpNet.DataAugmentation;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.MathTools;
using SharpNet.Networks;

namespace SharpNet.Datasets.PlumeLabs88;

public static class PlumeLabs88Utils
{
    public const string NAME = "PlumeLabs88";


    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(PlumeLabs88Utils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    //public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    public static string TrainDirectory => Path.Combine(WorkingDirectory, "Data", "x_train");
    public static string TestDirectory => Path.Combine(WorkingDirectory, "Data", "x_test");
    // ReSharper disable once MemberCanBePrivate.Global

    public static readonly int[] Raw_Shape_CHW = { 4, 128, 128};



    public const string ID_COLUMN_NAME = "ID";

    public static int RawValuesByElement => Utils.Product(Raw_Shape_CHW);

    public static DataFrame Load_YTrainPath()
    {
        return DataFrame.read_csv(YTrainPath, true, ColumnNameToType, true);
    }

    public static DataFrame Load_OutputTestRandomPath()
    {
        return DataFrame.read_csv(OutputTestRandomPath, true, ColumnNameToType, true);
    }
    

    public static void TransformPandasZipFiles(bool isTraining)
    {
        var maxValue = 0f;
        var minValue = float.MaxValue;
        for (int i = 0; i <= RawMaxId(isTraining); i += ROWS_BY_FILE)
        {
            var path = GetRawFileName(i, isTraining);
            Console.WriteLine($"Processing file {path}");
            var content = LoadZipFileContent(path);
            int rows = content.Length;
            var data = new short[rows * RawValuesByElement];
            void ProcessRow(int row)
            {
                var line = content[row];
                int idx = row * RawValuesByElement;
                var strings = line.Trim().Split(',');
                if (strings.Length != 1 + RawValuesByElement)
                {
                    throw new Exception($"Invalid line {line}");
                }
                foreach (var t in strings.Skip(1))
                {
                    var f = float.Parse(t);
                    minValue = Math.Min(minValue, f);
                    maxValue = Math.Max(maxValue, f);
                    data[idx++] = (short)Utils.NearestInt(f);
                }
            }
            Parallel.For(0, rows, ProcessRow);
            Utils.WriteBinaryFile(path.Replace(".csv.zip", ".bin"), data);
        }
        Console.WriteLine($"maxValue={maxValue}, minValue={minValue}, isTraining={isTraining}");
    }

    static float Normalize(float f, float mult, float mean)
    {
        return mult * (MathF.Log(1 + f) - mean);
    }
    public static void MeasureTimeToLoadAllRawData(bool isTraining)
    {
        var sw = Stopwatch.StartNew();
        for (int elementId = 0; elementId < RawMaxId(isTraining); ++elementId)
        {
            LoadRawElementId(elementId, isTraining);
        }
        ISample.Log.Info($"Loading all {RawMaxId(isTraining)} elements took {sw.ElapsedMilliseconds}ms");
    }

    public static void MakeLazyPredictions(bool isTraining)
    {
        var sw = Stopwatch.StartNew();
        var sb = new StringBuilder();
        sb.Append("Sep=," + Environment.NewLine);
        sb.Append("ID,TARGET" + Environment.NewLine);
        var datasetSample = new PlumeLabs88DatasetSample();
        datasetSample.TargetHeightAndWidth = 2;
        var elementShape = datasetSample.GetInputShapeOfSingleElement();
        var tensor = new CpuTensor<float>(elementShape);
        var span = tensor.SpanContent;
        for (int elementId = 0; elementId <= RawMaxId(isTraining); ++elementId)
        {
            datasetSample.LoadElementIdIntoSpan(elementId, span, isTraining);
            var avgLastChannel = tensor.RowSpanSlice(3, 1).ToArray().Select(datasetSample.UnNormalizeFeature).Average();
            int id = elementId* datasetSample.NumClass;
            for (int classId = 0; classId < datasetSample.NumClass; ++classId)
            {
                sb.Append($"{id++},{avgLastChannel.ToString(CultureInfo.InvariantCulture)}" + Environment.NewLine);
            }
        }
        ISample.Log.Info($"Making Lazy Predictions for {RawMaxId(isTraining)} elements took {sw.ElapsedMilliseconds}ms");
        var path = Path.Combine(WorkingDirectory, "dump", "lazy_pred_"+isTraining+".csv");
        File.WriteAllText(path, sb.ToString());
    }


    public static short[] LoadRawElementId(int elementId, bool isTraining)
    {
        var path = GetRawBinFileName(elementId, isTraining);
        var valuesByElement = RawValuesByElement;
        int firstElementId = FirstElementInFileContainingElementId(elementId);
        int positionInFile = (elementId - firstElementId) * valuesByElement;
        return Utils.ReadArrayFromBinaryFile<short>(path, positionInFile, valuesByElement);
    }


    public static string[] LoadZipFileContent(string path)
    {
        using var file = File.OpenRead(path);
        using var zip = new ZipArchive(file, ZipArchiveMode.Read);
        var entry = zip.Entries.First();
        using var sr = new StreamReader(entry.Open());
        string[] fileContent = sr.ReadToEnd().Split('\n').Where(t=>t.Length!=0).ToArray();
        return fileContent;
    }

    public static string YTrainPath => Path.Combine(WorkingDirectory, "Data", "y_train_lR55nNj.csv");
    public static string OutputTestRandomPath => Path.Combine(WorkingDirectory, "Data", "y_rand_HCYbSa3.csv");

    public static Dictionary<string, double> IdToPrediction(string path)
    {
        var res = new Dictionary<string, double>();
        var y_pred_df = DataFrame.read_csv(path, true, ColumnNameToType);
        var y_ID = y_pred_df.StringColumnContent("ID");
        var y_reod = y_pred_df.FloatColumnContent("TARGET");
        for (int i = 0; i < y_ID.Length; ++i)
        {
            res.Add(y_ID[i], y_reod[i]);
        }
        return res;
    }


    //?D public static int MaxId(bool isTrain) { return isTrain ? 99999 : 18148; }
    public const int RawMaxIdTraining = 99999;
    public const int RawMaxIdTest = 18148;
    public static int RawMaxId(bool isTrain) { return isTrain ? RawMaxIdTraining : RawMaxIdTest; }
    //public static int RawMaxId(bool isTrain) { return isTrain ? 19999: 18148; }
    
    public static void Run()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "log");

        ChallengeTools.Retrain(@"C:\Projects\Challenges\PlumeLabs88\dump", "543E24B90E", null, 0.9, false);

        //using var m = ModelAndDatasetPredictions.Load(@"C:\Projects\Challenges\PlumeLabs88\dump", "C8C8A2D3E2", true);
        //(_, _, DataFrame predictionsInTargetFormat, _, _) =  m.DatasetSample.ComputePredictionsAndRankingScoreV2(m.DatasetSample.TestDataset(), m.Model, false, true);
        //predictionsInTargetFormat.to_csv("c:/temp/toto.csv");

        //Misc.AdjustPlumeLabs88(@"C:\Projects\Challenges\PlumeLabs88\submit\84CBF5E063_predict_test_.csv");


        //ChallengeTools.ComputeAndSaveFeatureImportance(@"C:\Projects\Challenges\PlumeLabs88\Submit", "9B9DFA97F2_KFOLD_FULL");
        //ChallengeTools.EstimateLossContribution(@"C:\Projects\Challenges\PlumeLabs88\dump", "D885B11678");
        //ChallengeTools.Retrain(@"C:\Projects\Challenges\PlumeLabs88\dump", "9B9DFA97F2_KFOLD_FULL", null, 0.8, false);

        //ComputeStats();
        //Launch_HPO(10);
        //TransformPandasZipFiles(true);
        //MakeLazyPredictions(true);
        //MeasureTimeToLoadAllRawData(true);
        //MeasureTimeToLoadAllRawData(true);
        //BuildStatNormalizeFile();
        //ReorderColumns();
        //LaunchNeuralNetworkHPO();
        //LaunchLightGBMHPO(50, 2000);
        //LaunchCatBoostHPO(500, 1000);
        //LaunchSvmHPO();
        //TransformZipFiles(true);
        //string path = "c:/temp/toto.bin";
        //Utils.WriteBinaryFile(path, new short[] { 100, 0, 1, 2, 30, 4, 5, 6, 7, 8, 9 } );
        //        var res = Utils.ReadArrayFromFile<short>(path, 2, 4);
    }


    public const int ROWS_BY_FILE = 1000;

    private static void ComputeStats()
    {
        var accTotal = new DoubleAccumulator();
        var logAccTotal = new DoubleAccumulator();

        foreach (var isTrain in new[]{true, false})
        {
            var acc = new DoubleAccumulator();
            var logAcc = new DoubleAccumulator();
            for (int i = 0; i <= 2000 /*MaxId(isTrain)*/; i+= ROWS_BY_FILE)
            {
                var path = GetRawFileName(i, isTrain);
                var content = LoadZipFileContent(path);
                void ProcessLine(int row)
                {
                    var acc0 = new DoubleAccumulator();
                    var logAcc0 = new DoubleAccumulator();
                    var l = content[row];
                    foreach (var e in l.Split(',').Skip(1))
                    {
                        if (e.Length == 0)
                        {
                            continue;
                        }
                        var v = float.Parse(e);
                        acc.Add(v);
                        logAcc.Add(MathF.Log(1 + v));
                    }

                    lock (acc)
                    {
                        acc.Add(acc0);
                        logAcc.Add(logAcc0);
                    }
                        
                }
                Parallel.For(0, content.Length, ProcessLine);
            }
            Console.WriteLine($"isTrain={isTrain}, Acc={acc}, LogAcc={logAcc}");
            accTotal.Add(acc);
            logAccTotal.Add(logAcc);
        }
        Console.WriteLine($"FULL, accTotal={accTotal}, logAccTotal={logAccTotal}");
    }

    public static string GetRawFileName(int elementId, bool isTraining)
    {
        int start_id = elementId - elementId % ROWS_BY_FILE;
        int end_id = Math.Min(start_id + ROWS_BY_FILE - 1, RawMaxId(isTraining));
        return Path.Combine(GetDataDirectory(isTraining), $"{start_id}_{end_id}.csv.zip");
    }
    public static string GetRawBinFileName(int elementId, bool isTraining)
    {
        int start_id = FirstElementInFileContainingElementId(elementId);
        int end_id = Math.Min(start_id + PlumeLabs88Utils.ROWS_BY_FILE - 1, PlumeLabs88Utils.RawMaxId(isTraining));
        return Path.Combine(GetDataDirectory(isTraining), $"{start_id}_{end_id}.bin");
    }
    public static int FirstElementInFileContainingElementId(int elementId)
    {
        return elementId - elementId % PlumeLabs88Utils.ROWS_BY_FILE;
    }

    public static int RowInFile(int elementId)
    {
        return elementId % PlumeLabs88Utils.ROWS_BY_FILE;
    }

    public static string GetDataDirectory(bool isTraining) => isTraining ? PlumeLabs88Utils.TrainDirectory : PlumeLabs88Utils.TestDirectory;


    public static Type ColumnNameToType(string columnName)
    {
        if (columnName.Equals("ID"))
        {
            return typeof(string);
        }
        return typeof(float);
    }

       public static EfficientNetNetworkSample DefaultEfficientNetNetworkSample()
       {
           var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
               {
                   LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                   CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                   lambdaL2Regularization = 0.0005,
                   //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
                   NumEpochs = 10,
                   BatchSize = 1000,
                   InitialLearningRate = 0.01,

                   //Data augmentation
                   DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION,
                   WidthShiftRangeInPercentage = 0.0,
                   HeightShiftRangeInPercentage = 0.0,
                   HorizontalFlip = false,
                   VerticalFlip = false,
                   FillMode = ImageDataGenerator.FillModeEnum.Reflect,
                   AlphaMixup = 0.0,
                   AlphaCutMix = 0.0,
                   CutoutPatchPercentage = 0.0
               }
               .WithSGD(0.9, false)
               .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
           return config;

       }

    public static void Launch_HPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "log");
        var searchSpace = new Dictionary<string, object>
        {
            //{"KFold", 2},
            //{"PercentageInTraining", 0.9}, //will be automatically set to 1 if KFold is enabled
            {"PercentageInTraining", 0.9}, //will be automatically set to 1 if KFold is enabled
            { "BatchSize", new[] {100} },
            { "NumEpochs", new[] { numEpochs } },
            {"LossFunction", "Mse"},
            { "RandomizeOrder", true},

            //related to Dataset 
            { "NormalizeFeatureMean", new[] {/*0f,*/ 0.5f} },
            //{ "NormalizeFeatureMult", new[] { 1f, 0.8f, 1.2f} },
            //{ "NormalizeTargetMean", new[] {0f, 0.5f} },
            //{ "NormalizeTargetMult", new[] { 1f, 0.5f, 0.25f, 2f} },
            { "TargetHeightAndWidth", new[] {64} },
            
            // Optimizer 
            { "OptimizerType", new[] { "AdamW", } },
            //{ "OptimizerType", new[] { "SGD"} },
            { "AdamW_L2Regularization", new[] { 1e-5 /*, 1e-4, 1e-3, 1e-2, 1e-1*/ } }, //0.00001

            { "SGD_usenesterov", new[] { true, false } },
            //{ "lambdaL2Regularization", new[] { 0.0005, 0.001, 0.00005 } },
            { "lambdaL2Regularization", new[] {0.001, 0.0005, 0.0001, 0.00005 } }, // 0.0001 or 0.001
            //{"DefaultMobileBlocksDescriptionCount", new[]{5}},
            {"lastActivationLayer", nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)},

            // Learning Rate
            { "InitialLearningRate", new []{0.001, 0.01}}, //SGD: 0.01 //AdamW: 0.01 or 0.001
            // Learning Rate Scheduler
            //{ "LearningRateSchedulerType", new[] { "OneCycle" } },
            { "LearningRateSchedulerType", "CyclicCosineAnnealing" },
        };

        //var hpo = new GridSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new PlumeLabs88DatasetSample()), WorkingDirectory);
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new PlumeLabs88DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

}