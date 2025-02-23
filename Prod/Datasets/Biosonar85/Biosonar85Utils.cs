using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using SharpNet.LightGBM;
using SharpNet.Networks;
using SharpNet.Networks.Transformers;
using static SharpNet.Networks.NetworkSample;

namespace SharpNet.Datasets.Biosonar85;

public static class Biosonar85Utils
{
    public const string NAME = "Biosonar85";
    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");

    public static void Run()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, NAME, true);
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, NAME, true);

        //MixUp.DisplayStatsForAlphaMixUp();return;

        //var trainDataset = Load("X_train_23168_101_64_1024_512.bin", "Y_train.bin", true);
        //var testDataset = Load("X_test_950_101_64_1024_512.bin", "Y_test.bin", false);

        //var bin_file = Path.Combine(DataDirectory, "Y_train_ofTdMHi.csv.bin");
        //var tensor = CpuTensor<float>.LoadFromBinFile(bin_file, new[] { -1, 101, 64});
        //ParseLogFile(); return;

        //Log.Info(AllCombinations(0.4, 0.7));return;
        //Log.Info(AllCombinations(0.701, 0.85));return;

        ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "7BA5CBAEEE", null, percentageInTraining:0.5, retrainOnFullDataset:false, useAllAvailableCores:true);return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "transformers001", null, percentageInTraining:0.5, retrainOnFullDataset:false, useAllAvailableCores:true);return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "1EF57E45FC", null, percentageInTraining:0.5, retrainOnFullDataset:false, useAllAvailableCores:true);return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "FCB789043E_18", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "9811DDD19E_19", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "E5BA77E393", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;


        //ChallengeTools.ComputeAndSaveFeatureImportance(WorkingDirectory, "674CD08C52", true); return;
        //LaunchPyTorch();

        //ComputeAverage_avg();return;

        //Launch_HPO_spectrogram(20); return;

        //Launch_HPO_MEL_SPECTROGRAM_256_801_BinaryCrossentropy(20); return;
        // ?D Launch_HPO_MEL_SPECTROGRAM_256_801_BCEWithFocalLoss(20); return;
        //Launch_HPO_MEL_SPECTROGRAM_64_401(10); return;
        //Launch_HPO_MEL_SPECTROGRAM_SMALL_128_401(10); return;
        //Launch_HPO_MEL_SPECTROGRAM_128_401_BinaryCrossentropy(20); return;
        //Launch_HPO_MEL_SPECTROGRAM_128_401_BCEWithFocalLoss(10); return;

        //LaunchCatBoostHPO(1000); return;

        //LaunchLightGBMHPO(350);return;

        Launch_HPO_Transformers(10); return;
        //Launch_HPO(10);return;
        //OptimizeModel_AE0F13543C();
    }


    private static void LaunchPyTorch()
    {

        ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "PyTorch"), "DEE753FA84", null, percentageInTraining:0.5, retrainOnFullDataset:false, useAllAvailableCores:true);return;

    }


    public static void OptimizeModel_AE0F13543C()
    {
        var eachUpdate = new List<Tuple<string, string>>
                         {
                             Tuple.Create("HeightShiftRangeInPercentage","0.03"),
                             Tuple.Create("TopDropoutRate","0.2"),
                             Tuple.Create("HeightShiftRangeInPercentage","0.1"),
                             Tuple.Create("SkipConnectionsDropoutRate","0.5"),
                             Tuple.Create("SkipConnectionsDropoutRate","0.3"),
                             Tuple.Create("AlphaMixUp","1"),
                             Tuple.Create("InitialLearningRate","0.002"),
                             Tuple.Create("InitialLearningRate","0.003"),
                             Tuple.Create("CutoutPatchPercentage","0.1"),
                             Tuple.Create("CutoutPatchPercentage","0.2"),
                             Tuple.Create("RowsCutoutPatchPercentage","0.06"),
                             Tuple.Create("RowsCutoutPatchPercentage","0.18"),
                             //Tuple.Create("TopDropoutRate","0.4"), //done
                         };
        void ContentUpdate(string key, string newValue, IDictionary<string, string> content)
        {
            if (content.ContainsKey(key))
            {
                content[key] = newValue;
            }
        }

        foreach(var (key, newValue) in eachUpdate)
        {
            ISample.Log.Info($"Updating {key} to {newValue}");
            ChallengeTools.RetrainWithContentUpdate(Path.Combine(WorkingDirectory, "OptimizeModel", "1EF57E45FC_16"), "1EF57E45FC_16", a => ContentUpdate(key, newValue, a));
        }
    }

    // ReSharper disable once UnusedMember.Global
    public static string AllCombinations(double minPercentage, double maxPercentage)
    {
        var siteToCount = SiteToCount();

        double totalCount = siteToCount.Select(m => m.Value).Sum();
        List<Tuple<string, double>> allSites = new();
        foreach( var (site,count) in siteToCount)
        {
            allSites.Add(Tuple.Create(site,count/totalCount));
        }
        allSites = allSites.OrderByDescending(s => s.Item2).ToList();

        List<Tuple<string, double>> res = new();
        AllCombinations_Helper(0, 0, new List<string>(), allSites, res);
        res = res.Where(s => s.Item2 >= minPercentage && s.Item2 <= maxPercentage).ToList();
        var allAvailableCombinations = res.OrderBy(s=>s.Item2).Select(s=>s.Item1).ToList();
        return string.Join(", ", allAvailableCombinations);
    }

    private static Dictionary<string, int> SiteToCount()
    {
        var siteToCount = new Dictionary<string, int>();
        var ids = DataFrame.read_string_csv(Biosonar85DatasetSample.Y_train_path).StringColumnContent("id");
        foreach (var id in ids)
        {
            var site = IdToSite(id);
            if (!siteToCount.ContainsKey(site))
            {
                siteToCount[site] = 0;
            }
            ++siteToCount[site];
        }
        return siteToCount;
    }

    // ReSharper disable once UnusedMember.Local
    private static void ParseLogFile()
    {
        const string log_directory = @"C:\Projects\Challenges\Biosonar85\MandatorySitesForTraining";
        var csvPath = Path.Combine(log_directory, "data.csv");
        Dictionary<string, int> siteToCount = SiteToCount();
        foreach (var logPath in Directory.GetFiles(log_directory, "*.log"))
        {
            var lines = File.ReadAllLines(logPath);
            var currentBlock = new List<string>();

            for (int i = 0; i < lines.Length; ++i)
            {
                if (lines[i].Contains("sites for training"))
                {
                    ProcessBlock(currentBlock, siteToCount, csvPath);
                    currentBlock.Clear();
                    currentBlock.Add(lines[i]);
                }
                else
                {
                    currentBlock.Add(lines[i]);
                    if (i == lines.Length - 1)
                    {
                        ProcessBlock(currentBlock, siteToCount, csvPath);
                        currentBlock.Clear();
                    }
                }
            }
        }

    }
    private static void ProcessBlock(List<string> lines, Dictionary<string, int> siteToCount, string csvPath)
    {
        var sites = lines.FirstOrDefault(l => l.Contains("sites for training"));
        if (string.IsNullOrEmpty(sites))
        {
            return; 
        }
        sites = sites.Split().Last();
        double totalCount = siteToCount.Select(m => m.Value).Sum();
        double siteCount = 0;
        foreach (var site in sites.Split(','))
        {
            siteCount += siteToCount[site];
        }   
        var networkName = lines.FirstOrDefault(l => l.Contains("Network Name:"));
        if (string.IsNullOrEmpty(networkName))
        {
            return ;
        }
        networkName = networkName.Split().Last();
        int bestEpoch = -1;
        float bestScore = 0;
        for (int epoch = 1;; ++epoch)
        {
            var epochLine = lines.FirstOrDefault(l => l.Contains($"Epoch {epoch}/20"));
            if (string.IsNullOrEmpty(epochLine))
            {
                break;
            }
            const string toFind = "val_accuracy:";
            var index = epochLine.IndexOf(toFind, StringComparison.Ordinal);
            if (index < 0)
            {
                break;
            }
            var score = float.Parse(epochLine.Substring(index + toFind.Length).Trim().Split()[0]);
            if (bestEpoch == -1 || score > bestScore)
            {
                bestEpoch = epoch;
                bestScore = score;
            }
        }

        if (bestEpoch == -1)
        {
            return;
        }

        if (!File.Exists(csvPath))
        {
            File.AppendAllText(csvPath, "Sep=;"+Environment.NewLine);
            File.AppendAllText(csvPath, $"NetworkName;Sites;% in training;BestEpoch;BestScore" + Environment.NewLine);
        }
        File.AppendAllText(csvPath, $"{networkName};{sites};{siteCount / totalCount};{bestEpoch};{bestScore}" + Environment.NewLine);
    }



    private static void AllCombinations_Helper(int depth, double totalCount, List<string> currentPath, List<Tuple<string, double>> allSites, List<Tuple<string, double>> res)
    {
        if (depth>=allSites.Count)
        {
            res.Add(Tuple.Create("\""+string.Join(",", currentPath)+"\"", totalCount));
            return;
        }

        //without site at index 'depth'
        AllCombinations_Helper(depth + 1, totalCount, currentPath, allSites, res);

        //with site at index 'depth'
        currentPath.Add(allSites[depth].Item1);
        AllCombinations_Helper(depth + 1, totalCount+ allSites[depth].Item2, currentPath, allSites, res);
        currentPath.RemoveAt(currentPath.Count - 1);
    }

    public static string IdToSite(string id) 
    {
        return id.Split(new[] { '-', '.' })[1]; 
    }



    // ReSharper disable once UnusedMember.Global
    public static void ComputeAverage_avg()
    {
        var dfs = new List<DataFrame>();
        const string path = "C:/Projects/Challenges/Biosonar85/bbb";
        foreach (var file in new[]
                 {
                     @"09A06BF085_predict_test_.csv",
                     @"1EF57E45FC_16_predict_test_0,9637474614315744.csv",
                     @"1EF57E45FC_16_predict_test_0,9637474614315744.csv",
                     @"1EF57E45FC_16_predict_test_0,9637474614315744.csv",
                 })
        {
            dfs.Add(DataFrame.read_csv(Path.Combine(path, file), true, x => x == "id" ? typeof(string) : typeof(float)));
        }
        DataFrame.Average(dfs.ToArray()).to_csv(Path.Combine(path, "1EF57E45FC_16_09A06BF085.csv"));
    }


    public static (int[] shape, string n_fft, string hop_len, string f_min, string f_max, string top_db) ProcessXFileName(string xPath)
    {
        var xSplitted = Path.GetFileNameWithoutExtension(xPath).Split("_");
        var xShape = new[] { int.Parse(xSplitted[^8]), 1, int.Parse(xSplitted[^7]), int.Parse(xSplitted[^6]) };
        var n_fft = xSplitted[^5];
        var hop_len = xSplitted[^4];
        var f_min = xSplitted[^3];
        var f_max = xSplitted[^2];
        var top_db = xSplitted[^1];
        return (xShape, n_fft, hop_len, f_min, f_max, top_db);
    }

    public static InMemoryDataSet Load(string xFileName, [CanBeNull] string yFileNameIfAny, string csvPath, float mean, float stdDev, bool is_transformers_3d)
    {
        
        var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(mean, stdDev) };
        
        var xPath = Path.Join(DataDirectory, xFileName);
        (int[] xShape, var  _, var _, var _, var _, var _) = ProcessXFileName(xPath);

        ISample.Log.Info($"Loading {xShape[0]} tensors from {xPath} with shape {Tensor.ShapeToString(xShape)} (Y file:{yFileNameIfAny})");


        var xTensor = CpuTensor<float>.LoadFromBinFile(xPath, xShape);

        if (is_transformers_3d)
        {
            xTensor.ReshapeInPlace(xShape[0], xShape[2], xShape[3]);
            var xTensorNew = new CpuTensor<float>(new []{ xTensor.Shape[0], xTensor.Shape[2], xTensor.Shape[1] });
            xTensor.SwitchSecondAndThirdDimension(xTensorNew);
            xTensor = xTensorNew;
        }

        var yTensor = string.IsNullOrEmpty(yFileNameIfAny)
            ?null //no Y available for Dataset
            :CpuTensor<float>.LoadFromBinFile(Path.Join(DataDirectory, yFileNameIfAny), new []{ xShape[0], 1 });

        /*
        //mean = 0f; stdDev = 1f; //!D //no standardization
        // we disable standardization
        var xAccBefore = new DoubleAccumulator();
        xAccBefore.Add(xTensor.SpanContent);
        Log.Info($"Stats for {xFileName} before standardization: {xAccBefore}");

        //We standardize the input
        Log.Info($"Mean: {mean}, StdDev: {stdDev}");
        xTensor.LinearFunction(1f / stdDev, xTensor, -mean / stdDev);

        var xAccAfter = new DoubleAccumulator();
        xAccAfter.Add(xTensor.SpanContent);
        Log.Info($"Stats for {xFileName} after standardization: {xAccAfter}");
        */

        var yID = DataFrame.read_string_csv(csvPath).StringColumnContent("id");

        var dataset = new Biosonar85InMemoryDataSet(
            xTensor,
            yTensor,
            xFileName,
            meanAndVolatilityForEachChannel,
            yID);
        return dataset;
    }


    // ReSharper disable once RedundantAssignment
    // ReSharper disable once RedundantAssignment
    public static TensorListDataSet LoadTensorListDataSet(string xFileName, string xAugmentedFileNameIfAny, [CanBeNull] string yFileNameIfAny, string csvPath, float mean, float stdDev)
    {
        //!D we disable standardization
        mean = 0; stdDev = 1;

        var xPath = Path.Join(DataDirectory, xFileName);

        (int[] xShape, var _, var _, var _, var _, var _) = ProcessXFileName(xPath);

        List<CpuTensor<float>> xTensorList = CpuTensor<float>.LoadTensorListFromBinFileAndStandardizeIt(xPath, xShape , mean, stdDev);

        List<CpuTensor<float>> augmentedXTensorList = null;
        if (!string.IsNullOrEmpty(xAugmentedFileNameIfAny))
        {
            augmentedXTensorList = CpuTensor<float>.LoadTensorListFromBinFileAndStandardizeIt(Path.Join(DataDirectory, xAugmentedFileNameIfAny), xShape, mean, stdDev);
        }

        var yTensor = string.IsNullOrEmpty(yFileNameIfAny)
            ? null //no Y available for Dataset
            : CpuTensor<float>.LoadFromBinFile(Path.Join(DataDirectory, yFileNameIfAny), new[] { xShape[0], 1 });

        /*
        if (string.IsNullOrEmpty(yFileNameIfAny) && xShape[0] == 950)
        {
            var y_test_expected = DataFrame.read_csv(@"\\RYZEN2700X-DEV\Challenges\Biosonar85\Submit\2E3950406D_19_D560131427_45_avg_predict_test_0.956247550504151.csv", true, (col) => col == "id" ? typeof(string) : typeof(float));
            yTensor = new CpuTensor<float>(new[] { xShape[0], 1 }, y_test_expected.FloatColumnContent(y_test_expected.Columns[1]));
        }
        */

        /*
        ISample.Log.Info($"Standardization between -1 and +1");
        foreach (var t in xTensorList)
        {
            var xAccBefore = new MathTools.DoubleAccumulator();
            xAccBefore.Add(t.SpanContent);

            if (xAccBefore.Max > xAccBefore.Min)
            {
                mean = (float)xAccBefore.Average;
                var max = xAccBefore.Max-mean;
                var min = xAccBefore.Min-mean;
                var divider = (float) Math.Max(Math.Abs(min), Math.Abs(max));
                // x= (x-mean)/divider

                //We standardize the input between -1 and +1
                t.LinearFunction(1f / divider, t, -mean / divider);
            }
        }
        */

        var yID = DataFrame.read_string_csv(csvPath).StringColumnContent("id");

        var dataset = new TensorListDataSet(
            xTensorList,
            augmentedXTensorList,
            yTensor,
            xFileName,
            Objective_enum.Classification,
            null,
            null /* columnNames*/, 
            null, /* isCategoricalColumn */
            yID,
            "id",
            ',');
        return dataset;
    }


    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_Transformers(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //{ nameof(AbstractDatasetSample.PercentageInTraining), 0.5},
            { nameof(AbstractDatasetSample.PercentageInTraining), new[]{0.5}},
            
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.TRANSFORMERS_3D)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.85},

            //related to model
            //{ nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            //{ nameof(NetworkSample.BCEWithFocalLoss_Gamma), new []{0, /*0.35*/}},
            //{ nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            { nameof(NetworkSample.BatchSize), new[] {1024} },
            

            {nameof(TransformerNetworkSample.embedding_dim), 64},
            {nameof(TransformerNetworkSample.input_is_already_embedded), true },
            {nameof(TransformerNetworkSample.encoder_num_transformer_blocks), new[]{4} }, //4
            {nameof(TransformerNetworkSample.encoder_num_heads), new[]{8} },
            {nameof(TransformerNetworkSample.encoder_mha_use_bias_Q_K_V), new[]{false ,true } },
            {nameof(TransformerNetworkSample.encoder_mha_use_bias_O), true  }, // must be true
            {nameof(TransformerNetworkSample.encoder_mha_dropout), new[]{0.4 } }, //0.2 or 0.4
            {nameof(TransformerNetworkSample.encoder_feed_forward_dim), 4*64},
            {nameof(TransformerNetworkSample.encoder_feed_forward_dropout), new[]{ 0.2}}, //0.2
            {nameof(TransformerNetworkSample.encoder_is_causal), true}, //true
            {nameof(TransformerNetworkSample.output_shape_must_be_scalar), true},
            {nameof(TransformerNetworkSample.pooling_before_dense_layer), new[]{ nameof(POOLING_BEFORE_DENSE_LAYER.NONE) } }, //must be NONE
            {nameof(TransformerNetworkSample.layer_norm_before_last_dense), false}, // must be false

            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.weight_decay), new[]{0.0002}},

            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate),new[]{0.005},
            { nameof(NetworkSample.InitialLearningRate),new[]{0.07 }}, // 0.07 or 0.05

            // Learning Rate Scheduler
            //{nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle", "CyclicCosineAnnealing" } },
            {nameof(NetworkSample.LearningRateSchedulerType), "OneCycle" },
            //{nameof(NetworkSample.LearningRateSchedulerType), new[]{ "CyclicCosineAnnealing" } },
            {nameof(TransformerNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing),0.1},
            //{nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-6},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},

            
            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), new[]{0}},
            { nameof(NetworkSample.AlphaMixUp), new[]{1}},
            { nameof(NetworkSample.CutoutPatchPercentage), new[]{0, 0.1477559}},
            { nameof(NetworkSample.CutoutCount), new[]{0,1} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[]{0, 0.12661687}  },
            { nameof(NetworkSample.RowsCutoutCount), new[]{0, 1} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage), new[] { 0, 0.12661687 } },
           { nameof(NetworkSample.ColumnsCutoutCount), new[] { 0, 1} },
            //{ nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect)} },
            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Nearest)} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[] { 0} },
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[] {0} },
            


        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new TransformerNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //{"KFold", 2},
            {nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled
            
            //Related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.AUC)},
            { nameof(NetworkSample.BatchSize), new[] {256} },
            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.weight_decay), 0.01 },

            //Dataset
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.NETWORK_4D)},


            //{ "Use_MaxPooling", new[]{true,false}},
            //{ "Use_AvgPooling", new[]{/*true,*/false}}, //should be false
                

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0.5}, //must be > 0
            { nameof(NetworkSample.AlphaMixUp), new[] { 0 /*, 0.25*/} }, // must be 0
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1,0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.2 },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage), new[] {0.1, 0.2} },
            { nameof(NetworkSample.FillMode),nameof(ImageDataGenerator.FillModeEnum.Modulo) },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), 0.1 },

            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate), new []{0.01, 0.1 }}, //SGD: 0.01 //AdamW: 0.01 or 0.001
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperparameterSearchSpace.Range(0.001f,0.2f,AbstractHyperparameterSearchSpace.range_type.normal)},
            { nameof(NetworkSample.InitialLearningRate), 0.005}, 
            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            { nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
        };

        //var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        //var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
         var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new Biosonar85NetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_spectrogram(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            //{ nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.PNG_1CHANNEL)},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.PNG_1CHANNEL_V2)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {128} },
            
            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },
            
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" /*, "SGD"*/ } },
            //{ nameof(NetworkSample.nesterov), new[] { true, false } },
            { nameof(NetworkSample.weight_decay), new[] { /*0.005,0.001,, 0.05*/ 0.0005, 0.00025 } }, // to discard: 0.005, 0.05 0.001
            
            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 4 },

            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperparameterSearchSpace.Range(0.003f, 0.03f)},
            
            { nameof(NetworkSample.InitialLearningRate), new[]{0.0025, 0.005 , 0.01} }, //0.005 or 0.01

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} }, //discard: 0.4
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), new[] { 0.0,  1.0} }, //0
            { nameof(NetworkSample.AlphaMixUp), new[] { 0.0,  1.0} }, // 1 or 0 , 1 seems better
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {/*0,*/ 0.05, 0.1} }, //0.1 or 0.2
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {/*0 ,*/ 0.1} }, //0 or 0.1
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 }, // must be 0            
            { nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            
            //{ nameof(NetworkSample.VerticalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.Rotate180Degrees),new[]{true,false } },
            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            //{ nameof(NetworkSample.HeightShiftRangeInPercentage), AbstractHyperparameterSearchSpace.Range(0.05f, 0.30f) }, // must be > 0 , 0.1 seems good default
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0.05, 0.1   } }, //to discard: 0.2
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}}, // must be 0

        };

        //model: FB0927A468_17
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.weight_decay)] = 0.00025;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixUp)] = 1;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.0025;
        searchSpace[nameof(NetworkSample.HorizontalFlip)] = true;
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        searchSpace[nameof(NetworkSample.num_epochs)] = 20;
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.93;


        //!D
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = HyperparameterSearchSpace.Range(0.025f, 0.075f);
        searchSpace[nameof(NetworkSample.weight_decay)] = HyperparameterSearchSpace.Range(0.0002f, 0.0003f);
        searchSpace[nameof(NetworkSample.AlphaMixUp)] = HyperparameterSearchSpace.Range(0.75f, 1.25f);
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = HyperparameterSearchSpace.Range(0.002f, 0.03f);
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = HyperparameterSearchSpace.Range(0.05f, 0.15f);
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = HyperparameterSearchSpace.Range(0.05f, 0.15f);
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.94;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = new[] { -1,5,6 };


        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_SMALL_128_401(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
     

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 1.0}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), false},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_SMALL_128_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.93},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), 32}, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.num_epochs), 1},

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), false},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.weight_decay), 0.0005 },

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 2 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), 0.01},

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} },
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixUp), 0},
            //{ nameof(NetworkSample.CutoutPatchPercentage), 0.0 },
            //{ nameof(NetworkSample.CutoutCount), 0 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage), 0.2},
            { nameof(NetworkSample.RowsCutoutCount), 0},
            { nameof(NetworkSample.RowsCutoutPatchPercentage),  0.0 },
            { nameof(NetworkSample.HorizontalFlip),true},
            //{ nameof(NetworkSample.VerticalFlip),true},

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), 0.0},
            { nameof(NetworkSample.WidthShiftRangeInPercentage), 0.0},

        };

     
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_128_401_BinaryCrossentropy(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        //with Data Augmentation on waveform
        var searchSpace = new Dictionary<string, object>
        {
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_128_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)},
            { nameof(NetworkSample.BatchSize), new[] {128} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.weight_decay), new[] { 0.000125, 0.00025, 0.0005 } },

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), new[] { 5, -1 } },

            { nameof(EfficientNetNetworkSample.TopDropoutRate), new[] { 0f, 0.2f, 0.5f }},
            { nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate), new[] { 0f, 0.2f, 0.5f } },


            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new[] { 0.005, 0.01, 0.02} },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[] { 0, 0.1 } },
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            //{nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixUp), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] { 0, 0.05, 0.1} },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] { 0, 0.05, 0.1 } },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            //{ nameof(NetworkSample.HorizontalFlip),true/*new[]{true,false } */},

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05} },
            //{ nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };
        //best score:  0.9469099 (FCB789043E)
        //AdamW weight_decay = 0.0005
        //CutoutPatchPercentage = 0.05
        //DefaultMobileBlocksDescriptionCount = 5
        //HeightShiftRangeInPercentage = 0.05
        //InitialLearningRate = 0.01
        //OneCycle_PercentInAnnealing = 0.1
        //RowsCutoutPatchPercentage = 0.1
        //SkipConnectionsDropoutRate = 0
        //TopDropoutRate = 0.5

        //TO find the best learning rate
        //searchSpace[nameof(NetworkSample.BatchSize)] = 8;
        //searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 1.0;

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_128_401_BCEWithFocalLoss(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        //when LossFunction = BCEWithFocalLoss && PercentageInTraining == 0.5
        //  InitialLearningRate == 0.01
        //  AdamW weight_decay == 0.0005
        //  AlphaMixUp == 1.2

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_128_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.93},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {128} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.weight_decay), new[] {0.00015,  0.00025,  0.0005} },

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new[]{0.00125, 0.0025, 0.005 } },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} },
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixUp), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0.1, 0.2} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),true/*new[]{true,false } */},

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };




        searchSpace[nameof(NetworkSample.weight_decay)] = 0.0005;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixUp)] = 1.2; //new[] { 1, 1.2, 1.5 }; //1.2; //1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = 0.7; //new[] { 0, 0.7 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass)] = 0.5; //new[] { 0.4, 0.5, 0.6 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.01; //new[] { 0.001, 0.005, 0.01 };
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss); //new[]{nameof(EvaluationMetricEnum.BinaryCrossentropy), nameof(EvaluationMetricEnum.BCEWithFocalLoss)};
        searchSpace[nameof(NetworkSample.num_epochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0.1; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = 0.4f; // new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = 0.4f; //new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_256_801_BinaryCrossentropy(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled
            
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_256_801)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {42} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.weight_decay),  new[]{0.0000625,0.000125,0.00025}},

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 6 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new[]{0.00125,0.0025,0.005}},

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing),0.1},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            //{nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},

            {nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate),0.4},
            {nameof(EfficientNetNetworkSample.TopDropoutRate), 0.4},

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixUp), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), 0.1477559 },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.12661687 },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            { nameof(NetworkSample.HorizontalFlip), new[]{true,false } },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), 0.06299627 },
            //{ nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };
        //best results 0.95572406 (2D1DD70DA6)
        //AdamW weight_decay = 0.000125
        //HorizontalFlip = False
        //InitialLearningRate = 0.0025
        //for: 07D3470DB7_16, Valid Score: 0.9546, Test score AUC: 0.938326148145509
   
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_256_801_BCEWithFocalLoss(int num_epochs = 20, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled
            
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_256_801)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.95},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), new []{0, 0.35}},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            { nameof(NetworkSample.BatchSize), new[] {42} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.weight_decay),  0.000125},

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 6},

            //{ nameof(EfficientNetNetworkSample.DefaultActivation), new[]{nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU),nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_CLIPPED_RELU),nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH)   }},
            

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate),new[]{0.0025, 0.0035}},

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing),0.1},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            //{nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},

            {nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate),0.4},
            {nameof(EfficientNetNetworkSample.TopDropoutRate),new[]{0.4,0.2}},

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixUp), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), 0.1477559 },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[]{0.12661687,0.25}  },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            //{ nameof(NetworkSample.HorizontalFlip), false /*new[]{true,false }*/ },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0.06299627, 0.15} },

        };

        //like 1EF57E45FC_16
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = 0.35;
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = 0.3;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.12661687;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.06299627;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.25;

        //changes on 1EF57E45FC_16
        searchSpace[nameof(NetworkSample.num_epochs)] = new[]{/*50,100*/ 50};
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[] { /*0.001, */0.001};
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = 0.75; //new[] { 0.4, 0.5, 0.6, 0.65}; //0.4
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = 0.5; //new[] { 0.3, 0.5, 0.6, 0.65 }; //0.3



        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }




    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_64_401(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_64_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.90},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},


            { nameof(NetworkSample.BatchSize), new[] {128} },

            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.weight_decay), new[] {0.00015,  0.00025,  0.0005} },
            
            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), 0.05 },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} },
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixUp), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutCount), new[] {1, 3, 5} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),new[]{true,false } },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05, 0.10} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    

    /// <summary>
    /// The default EfficientNet Hyperparameters for CIFAR10
    /// </summary>
    /// <returns></returns>
    private static EfficientNetNetworkSample DefaultEfficientNetNetworkSample()
    {
        var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
        {
            LossFunction = EvaluationMetricEnum.BinaryCrossentropy,
            EvaluationMetrics = new List<EvaluationMetricEnum> {EvaluationMetricEnum.AUC},
            CompatibilityMode = CompatibilityModeEnum.TensorFlow,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
            num_epochs = 10,
            BatchSize = 64,
            //Data augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            //AlphaMixUp = 0.0,
            //AlphaCutMix = 0.0,
            //CutoutPatchPercentage = 0.0
        }
            .WithSGD(lr:0.1, 0.9, weight_decay: 0.0005, false)
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
        return config;

    }



    // ReSharper disable once UnusedMember.Local
    private static void LaunchCatBoostHPO(int iterations = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local
        var searchSpace = new Dictionary<string, object>
          {

              //related to Dataset 
              //{"KFold", 2},
              { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled
              { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
              { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.LIBROSA_FEATURES)},
              //{ nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.92},

              //related to model
              {"loss_function", nameof(CatBoostSample.loss_function_enum.Logloss)},
              {"eval_metric", nameof(CatBoostSample.metric_enum.Accuracy)},
              { "thread_count",1},
              { "task_type","GPU"},
              { "logging_level", nameof(CatBoostSample.logging_level_enum.Verbose)},
              { "allow_writing_files",false},
              { "iterations", iterations },
              //{ "od_type", "Iter"},
              //{ "od_wait",iterations/10},
              { "depth", new[]{5,6,7,8,9,10} },
              { "learning_rate", new[]{0.01,0.02, 0.03}},
              //{ "random_strength",AbstractHyperparameterSearchSpace.Range(1e-9f, 10f, AbstractHyperparameterSearchSpace.range_type.normal)},
              //{ "bagging_temperature",AbstractHyperparameterSearchSpace.Range(0.0f, 2.0f)},
              { "l2_leaf_reg",new[]{0,1,5,10,20}},
              //{"grow_policy", new []{ "SymmetricTree", "Depthwise" /*, "Lossguide"*/}},
          };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample{loss_function = CatBoostSample.loss_function_enum.Logloss , eval_metric = CatBoostSample.metric_enum.Accuracy }, new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Local
    private static void LaunchLightGBMHPO(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.LIBROSA_FEATURES)},
            //{ nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.92},
          
            //uncomment appropriate one
            ////for regression:
            //{"objective", "regression"},      

            ////for binary classification:
            {"objective", "binary"},
            {"metric", "accuracy"},

            ////for multi class classification:
            //{"objective", "multiclass"},      
            //{"num_class", number_of_class },

            //high priority
            
            //!D { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_fraction", 0.8 },
            //!D
            { "bagging_freq", 5 },

            //{ "boosting", new []{"gbdt", "dart"}},
            { "boosting", new []{"dart"}},
            
            //!D { "colsample_bytree",AbstractHyperparameterSearchSpace.Range(0.3f, 1.0f)},
            { "colsample_bytree",0.8},

            //{ "early_stopping_round", num_iterations/10 },


            //against over fitting
            { "extra_trees", new[] { true , false } },
            { "min_sum_hessian_in_leaf", new[]{0,0.1,1.0} },
            { "path_smooth", new[]{0,0.1,1.0} },
            { "min_gain_to_split", new[]{0,0.1,1.0} },
            { "max_bin", new[]{10,50, 255} },
            { "lambda_l1",new[]{0,1,5,10,50}},
            { "lambda_l2",new[]{0,1,5,10,50}},
            { "min_data_in_leaf", new[]{10,100,500} },
            
            
            //!D { "learning_rate",AbstractHyperparameterSearchSpace.Range(0.005f, 0.1f)},
            { "learning_rate", new[]{0.001, 0.005, 0.01, 0.05, 0.1}},
            
            //!D{ "max_depth", new[]{10, 20, 50, 100, 255} },
            { "max_depth", new[]{10, 50} },

            { "num_iterations", num_iterations },
            { "num_leaves", new[]{3, 10, 50} },
            { "num_threads", -1},
            { "verbosity", "0" },

            ////medium priority
            //{ "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            //{ "lambda_l2",AbstractHyperparameterSearchSpace.Range(0f, 2f)},
            //{ "min_data_in_bin", new[]{3, 10, 100, 150}  },
            //{ "max_bin", AbstractHyperparameterSearchSpace.Range(10, 255) },
            //{ "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode
            //{ "skip_drop",AbstractHyperparameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            ////low priority
            ////{ "colsample_bynode",AbstractHyperparameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            //{ "path_smooth", AbstractHyperparameterSearchSpace.Range(0f, 1f) }, //low priority
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample{objective =  LightGBMSample.objective_enum.binary}, new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

}