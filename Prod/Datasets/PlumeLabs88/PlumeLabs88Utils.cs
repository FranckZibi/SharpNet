using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpNet.CPU;
using SharpNet.DataAugmentation;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using SharpNet.MathTools;
using SharpNet.Networks;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Datasets.PlumeLabs88;

public static class PlumeLabs88Utils
{
    public const string NAME = "PlumeLabs88";
    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string TrainDirectory => Path.Combine(WorkingDirectory, "Data", "x_train");
    private static string TestDirectory => Path.Combine(WorkingDirectory, "Data", "x_test");
    // ReSharper disable once MemberCanBePrivate.Global
    public static readonly int[] Raw_Shape_CHW = { 4, 128, 128};
    private const int ROWS_BY_FILE = 1000;
    public const int RawMaxIdTraining = 99999;
    public const int RawMaxIdTest = 18148;
    public const string ID_COLUMN_NAME = "ID";


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
    public static float Normalize(float f, float mult, float mean)
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
        var elementShape = datasetSample.X_Shape(1).Skip(1).ToArray();
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
    private static string[] LoadZipFileContent(string path)
    {
        using var file = File.OpenRead(path);
        using var zip = new ZipArchive(file, ZipArchiveMode.Read);
        var entry = zip.Entries.First();
        using var sr = new StreamReader(entry.Open());
        string[] fileContent = sr.ReadToEnd().Split('\n').Where(t=>t.Length!=0).ToArray();
        return fileContent;
    }
    private static string YTrainPath => Path.Combine(WorkingDirectory, "Data", "y_train_lR55nNj.csv");
    private static string OutputTestRandomPath => Path.Combine(WorkingDirectory, "Data", "y_rand_HCYbSa3.csv");
    private static int RawMaxId(bool isTrain) { return isTrain ? RawMaxIdTraining : RawMaxIdTest; }
    public static void Run()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, NAME);
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, NAME);
        //ChallengeTools.EstimateLossContribution(@"C:\Projects\Challenges\PlumeLabs88\dump", "D885B11678");
        ChallengeTools.Retrain(@"C:\Projects\Challenges\PlumeLabs88\dump", "8D8FEBFE79", n_splits: null, percentageInTraining: 0.9, retrainOnFullDataset: false);

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
    public static void ComputeStats()
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
    public static void Launch_HPO(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, NAME);
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, NAME);
        var searchSpace = new Dictionary<string, object>
        {
            
            //related to Dataset 
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            { "NormalizeFeatureMean", new[] {/*0f,*/ 0.5f} },
            //{ "NormalizeFeatureMult", new[] { 1f, 0.8f, 1.2f} },
            //{ "NormalizeTargetMean", new[] {0f, 0.5f} },
            //{ "NormalizeTargetMult", new[] { 1f, 0.5f, 0.25f, 2f} },
            { "TargetHeightAndWidth", new[] {64} },


            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.Mse)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Mse)},
            //{ "KFold", 2},
            //{ nameof(AbstractDatasetSample.PercentageInTraining), 0.9}, //will be automatically set to 1 if KFold is enabled
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.9}, //will be automatically set to 1 if KFold is enabled
            { nameof(NetworkSample.BatchSize), new[] {100} },
            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW", } },
            //{ nameof(NetworkSample.OptimizerType), new[] { "SGD"} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] { 1e-5 /*, 1e-4, 1e-3, 1e-2, 1e-1*/ } }, //0.00001
            { nameof(NetworkSample.SGD_usenesterov), new[] { true, false } },
            //{ nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005, 0.001, 0.00005 } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] {0.001, 0.0005, 0.0001, 0.00005 } }, // 0.0001 or 0.001
            //{nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), new[]{5}},
            {"LastActivationLayer", nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)},
            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new []{0.001, 0.01}}, //SGD: 0.01 //AdamW: 0.01 or 0.001
            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            { nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
        };
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new PlumeLabs88DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    private static string GetRawFileName(int elementId, bool isTraining)
    {
        int start_id = elementId - elementId % ROWS_BY_FILE;
        int end_id = Math.Min(start_id + ROWS_BY_FILE - 1, RawMaxId(isTraining));
        return Path.Combine(GetDataDirectory(isTraining), $"{start_id}_{end_id}.csv.zip");
    }
    private static string GetRawBinFileName(int elementId, bool isTraining)
    {
        int start_id = FirstElementInFileContainingElementId(elementId);
        int end_id = Math.Min(start_id + ROWS_BY_FILE - 1, RawMaxId(isTraining));
        return Path.Combine(GetDataDirectory(isTraining), $"{start_id}_{end_id}.bin");
    }
    private static int FirstElementInFileContainingElementId(int elementId)
    {
        return elementId - elementId % ROWS_BY_FILE;
    }
    private static string GetDataDirectory(bool isTraining) => isTraining ? TrainDirectory : TestDirectory;
    public static Type ColumnNameToType(string columnName)
    {
        if (columnName.Equals("ID"))
        {
            return typeof(string);
        }
        return typeof(float);
    }
    private static EfficientNetNetworkSample DefaultEfficientNetNetworkSample()
       {
           var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
               {
                   LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                   CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                   lambdaL2Regularization = 0.0005,
                   //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
                   num_epochs = 10,
                   BatchSize = 1000,
                   InitialLearningRate = 0.01,

                   //Data augmentation
                   DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION,
                   WidthShiftRangeInPercentage = 0.0,
                   HeightShiftRangeInPercentage = 0.0,
                   HorizontalFlip = false,
                   VerticalFlip = false,
                   FillMode = ImageDataGenerator.FillModeEnum.Reflect,
                   AlphaMixUp = 0.0,
                   AlphaCutMix = 0.0,
                   CutoutPatchPercentage = 0.0
               }
               .WithSGD(0.9, false)
               .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
           return config;

       }
    private static int RawValuesByElement => Utils.Product(Raw_Shape_CHW);
}
