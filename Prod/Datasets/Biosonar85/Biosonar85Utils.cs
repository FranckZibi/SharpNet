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

        //Mixup.DisplayStatsForAlphaMixup();return;

        //var trainDataset = Load("X_train_23168_101_64_1024_512.bin", "Y_train_23168_1_64_1024_512.bin", true);
        //var testDataset = Load("X_test_950_101_64_1024_512.bin", "Y_test_950_1_64_1024_512.bin", false);

        //var bin_file = Path.Combine(DataDirectory, "Y_train_ofTdMHi.csv.bin");
        //var tensor = CpuTensor<float>.LoadFromBinFile(bin_file, new[] { -1, 101, 64});
        //ParseLogFile(); return;

        //Log.Info(AllCombinations(0.4, 0.7));return;
        //Log.Info(AllCombinations(0.701, 0.85));return;

        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "2A4F619211", null, percentageInTraining:0.5, retrainOnFullDataset:false, useAllAvailableCores:true);return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "FCB789043E_18", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "9811DDD19E_19", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "E5BA77E393", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;


        //ChallengeTools.ComputeAndSaveFeatureImportance(WorkingDirectory, "674CD08C52", true); return;


        //ComputeAverage_avg();return;

        //Launch_HPO_spectrogram(20); return;

        //Launch_HPO_MEL_SPECTROGRAM_256_801_BinaryCrossentropy(20); return;
        Launch_HPO_MEL_SPECTROGRAM_256_801_BCEWithFocalLoss(20); return;
        //Launch_HPO_MEL_SPECTROGRAM_64_401(10); return;
        //Launch_HPO_MEL_SPECTROGRAM_SMALL_128_401(10); return;
        //Launch_HPO_MEL_SPECTROGRAM_128_401_BinaryCrossentropy(20); return;
        //Launch_HPO_MEL_SPECTROGRAM_128_401_BCEWithFocalLoss(10); return;

        //LaunchCatBoostHPO(1000); return;

        //LaunchLightGBMHPO(350);return;

        //Launch_HPO_Transformers(10); return;
        //Launch_HPO(10);return;
        //OptimizeModel_AE0F13543C();
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
                             Tuple.Create("AlphaMixup","1"),
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
        const string path = @"\\RYZEN2700X-DEV\Challenges\Biosonar85\Submit\";
        foreach (var file in new[]
                 {
                     @"7E45F84676_predict_test_0,9353867531264475.csv",
                     @"569C5C14D2_predict_test_0.936063704706595.csv",
                 })
        {
            dfs.Add(DataFrame.read_csv(Path.Combine(path, file), true, x => x == "id" ? typeof(string) : typeof(float)));
        }
        DataFrame.Average(dfs.ToArray()).to_csv(Path.Combine(path, "7E45F84676_569C5C14D2_avg.csv"));
    }


    public static (int[] shape, string n_fft, string hop_len, string f_min, string f_max, string top_db) ProcessXFileName(string xPath)
    {
        var xSplitted = Path.GetFileNameWithoutExtension(xPath).Split("_");
        var xShape = new[] { int.Parse(xSplitted[^8]), int.Parse(xSplitted[^7]), int.Parse(xSplitted[^6]) };
        var n_fft = xSplitted[^5];
        var hop_len = xSplitted[^4];
        var f_min = xSplitted[^3];
        var f_max = xSplitted[^2];
        var top_db = xSplitted[^1];
        return (xShape, n_fft, hop_len, f_min, f_max, top_db);
    }

    public static InMemoryDataSet Load(string xFileName, [CanBeNull] string yFileNameIfAny, string csvPath, float mean = 0f, float stdDev = 1f)
    {
        
        var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(mean, stdDev) };
        
        var xPath = Path.Join(DataDirectory, xFileName);
        (int[] xShape, var  _, var _, var _, var _, var _) = ProcessXFileName(xPath);

        ISample.Log.Info($"Loading {xShape[0]} tensors from {xPath} with shape {Tensor.ShapeToString(xShape)} (Y file:{yFileNameIfAny})");


        var xTensor = CpuTensor<float>.LoadFromBinFile(xPath, xShape);
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
    private static void Launch_HPO_Transformers(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //{ nameof(AbstractDatasetSample.PercentageInTraining), 0.5},
            { nameof(AbstractDatasetSample.PercentageInTraining), new[]{0.5,0.8}},
            
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.TRANSFORMERS_3D)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.87},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), new []{0, /*0.35*/}},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            //{ nameof(NetworkSample.BatchSize), new[] {1024} },
            { nameof(NetworkSample.BatchSize), new[] {256} },

            {nameof(TransformerNetworkSample.embedding_dim), 64},
            {nameof(TransformerNetworkSample.input_is_already_embedded), true },
            {nameof(TransformerNetworkSample.encoder_num_transformer_blocks), new[]{4} },
            {nameof(TransformerNetworkSample.encoder_num_heads), new[]{8} },
            {nameof(TransformerNetworkSample.encoder_mha_use_bias_Q_V_K), new[]{false /*,true*/ } },
            {nameof(TransformerNetworkSample.encoder_mha_use_bias_O), true  }, // must be true
            {nameof(TransformerNetworkSample.encoder_mha_dropout), new[]{0.2, 0.4 } },
            {nameof(TransformerNetworkSample.encoder_feed_forward_dim), 4*64},
            {nameof(TransformerNetworkSample.encoder_feed_forward_dropout), new[]{ 0.2,0.4 }}, //0.2
            {nameof(TransformerNetworkSample.encoder_use_causal_mask), true},
            {nameof(TransformerNetworkSample.output_shape_must_be_scalar), true},
            {nameof(TransformerNetworkSample.pooling_before_dense_layer), new[]{ nameof(POOLING_BEFORE_DENSE_LAYER.NONE) /*,nameof(POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling), nameof(POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)*/ } }, //must be NONE
            {nameof(TransformerNetworkSample.layer_norm_before_last_dense), false}, // must be false

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            //{ nameof(NetworkSample.lambdaL2Regularization), 0.0005 },
            { nameof(NetworkSample.lambdaL2Regularization), 0},
            
            { nameof(NetworkSample.AdamW_L2Regularization), 0.00005},
            //{ nameof(NetworkSample.AdamW_L2Regularization), new[]{0.00001,0.00005, 0.0001,0.0005,0.001, 0.005,0.01}},

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate),0.005},
            //{ nameof(NetworkSample.InitialLearningRate),new[]{0.0001,0.0005,0.001, 0.005, 0.01, 0.05, 0.1}},

            // Learning Rate Scheduler
            //{nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle", "CyclicCosineAnnealing" } },
            {nameof(NetworkSample.LearningRateSchedulerType), "OneCycle" },
            //{nameof(NetworkSample.LearningRateSchedulerType), new[]{ "CyclicCosineAnnealing" } },
            {nameof(TransformerNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing),0.1},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-6},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},

            
            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), new[]{0,0.5215608}},
            { nameof(NetworkSample.AlphaMixup), new[]{0,1.2}},
            { nameof(NetworkSample.CutoutPatchPercentage), new[]{0,0.06450188,0.1477559}},
            { nameof(NetworkSample.CutoutCount), 1 },
            //{ nameof(NetworkSample.RowsCutoutPatchPercentage), new[]{0.12661687}  },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[]{ 0, 0.047216333 }  },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage), new[] {  0, 0.007914994} },
            { nameof(NetworkSample.ColumnsCutoutCount), new[] { 1} },
            //{ nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect)} },
            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Nearest)} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[] { 0, 0.034252543 } },
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[] {0, 0.06299627, 0.008304262 } },
            


        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new TransformerNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //{"KFold", 2},
            {nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled
            
            //Related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.AUC)},
            { nameof(NetworkSample.BatchSize), new[] {256} },
            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            //{ nameof(NetworkSample.OptimizerType), new[] { "SGD"} },
            //{ nameof(NetworkSample.AdamW_L2Regularization), AbstractHyperparameterSearchSpace.Range(0.00001f,0.01f, AbstractHyperparameterSearchSpace.range_type.normal) },
            //{ nameof(NetworkSample.AdamW_L2Regularization), AbstractHyperparameterSearchSpace.Range(0.00001f,0.01f, AbstractHyperparameterSearchSpace.range_type.normal) },
            { nameof(NetworkSample.AdamW_L2Regularization), 0.01 },

            //Dataset
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.NETWORK_4D)},


            //{ "Use_MaxPooling", new[]{true,false}},
            //{ "Use_AvgPooling", new[]{/*true,*/false}}, //should be false
                

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0.5}, //must be > 0
            { nameof(NetworkSample.AlphaMixup), new[] { 0 /*, 0.25*/} }, // must be 0
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1,0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.2 },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage), new[] {0.1, 0.2} },
            //{ nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.VerticalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.Rotate180Degrees),new[]{true,false } },
            { nameof(NetworkSample.FillMode),nameof(ImageDataGenerator.FillModeEnum.Modulo) },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), 0.1 },
            //{ nameof(NetworkSample.HeightShiftRangeInPercentage), new[] { 0.0 , 0.1,0.2 } }, //0
            //{ nameof(NetworkSample.ZoomRange), new[] { 0.0 , 0.05 } },

            

            //{ nameof(NetworkSample.SGD_usenesterov), new[] { true, false } },
            //{ nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005, 0.001, 0.00005 } },
            //{ nameof(NetworkSample.lambdaL2Regularization), new[] {0.001, 0.0005, 0.0001, 0.00005 } }, // 0.0001 or 0.001
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), new[]{5}},
            //{ nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},

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
    private static void Launch_HPO_spectrogram(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
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
            
            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },
            
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" /*, "SGD"*/ } },
            //{ nameof(NetworkSample.SGD_usenesterov), new[] { true, false } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005 /*, 0.001*/} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] { /*0.005,0.001,, 0.05*/ 0.0005, 0.00025 } }, // to discard: 0.005, 0.05 0.001
            
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
            { nameof(NetworkSample.AlphaMixup), new[] { 0.0,  1.0} }, // 1 or 0 , 1 seems better
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
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.0025;
        searchSpace[nameof(NetworkSample.HorizontalFlip)] = true;
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.93;


        //!D
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = HyperparameterSearchSpace.Range(0.025f, 0.075f);
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = HyperparameterSearchSpace.Range(0.0002f, 0.0003f);
        searchSpace[nameof(NetworkSample.AlphaMixup)] = HyperparameterSearchSpace.Range(0.75f, 1.25f);
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = HyperparameterSearchSpace.Range(0.002f, 0.03f);
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = HyperparameterSearchSpace.Range(0.05f, 0.15f);
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = HyperparameterSearchSpace.Range(0.05f, 0.15f);
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.94;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = new[] { -1,5,6 };


        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,StEUS", "BON,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,ARUBA", "StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "BON,JAM,ARUBA,StEUS", "BON,StMARTIN", "GUA,StEUS,BERMUDE", "GUA,StEUS,BAHAMAS", "GUA,ARUBA,BERMUDE", "GUA,ARUBA,BAHAMAS", "StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "BON,JAM,ARUBA,StEUS,BERMUDE", "BON,JAM,ARUBA,StEUS,BAHAMAS", "BON,StMARTIN,BERMUDE", "BON,StMARTIN,BAHAMAS", "GUA,StEUS,BAHAMAS,BERMUDE", "GUA,ARUBA,BAHAMAS,BERMUDE", "BON,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,ARUBA,StEUS", "BON,StMARTIN,BAHAMAS,BERMUDE", "BON,StMARTIN,StEUS", "BON,StMARTIN,ARUBA", "GUA,ARUBA,StEUS,BERMUDE", "GUA,ARUBA,StEUS,BAHAMAS", "BON,StMARTIN,StEUS,BERMUDE", "BON,StMARTIN,StEUS,BAHAMAS", "BON,StMARTIN,ARUBA,BERMUDE", "BON,StMARTIN,ARUBA,BAHAMAS", "GUA,ARUBA,StEUS,BAHAMAS,BERMUDE", "BON,StMARTIN,StEUS,BAHAMAS,BERMUDE", "BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "BON,StMARTIN,ARUBA,StEUS", "GUA,JAM", "BON,StMARTIN,ARUBA,StEUS,BERMUDE", "BON,StMARTIN,ARUBA,StEUS,BAHAMAS", "GUA,JAM,BERMUDE", "GUA,JAM,BAHAMAS", "BON,StMARTIN,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,JAM,BAHAMAS,BERMUDE", "GUA,JAM,StEUS", "GUA,JAM,ARUBA", "BON,StMARTIN,JAM", "GUA,JAM,StEUS,BERMUDE", "GUA,JAM,StEUS,BAHAMAS", "GUA,JAM,ARUBA,BERMUDE", "GUA,JAM,ARUBA,BAHAMAS", "BON,StMARTIN,JAM,BERMUDE", "BON,StMARTIN,JAM,BAHAMAS", "GUA,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,JAM,ARUBA,StEUS", "BON,StMARTIN,JAM,BAHAMAS,BERMUDE", "GUA,StMARTIN", "BON,StMARTIN,JAM,StEUS", "BON,StMARTIN,JAM,ARUBA", "GUA,JAM,ARUBA,StEUS,BERMUDE", "GUA,JAM,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,BERMUDE", "GUA,StMARTIN,BAHAMAS", "GUA,BON", "BON,StMARTIN,JAM,StEUS,BERMUDE", "BON,StMARTIN,JAM,StEUS,BAHAMAS", "BON,StMARTIN,JAM,ARUBA,BERMUDE", "BON,StMARTIN,JAM,ARUBA,BAHAMAS", "GUA,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,BAHAMAS,BERMUDE", "GUA,BON,BERMUDE", "GUA,BON,BAHAMAS", "BON,StMARTIN,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,StEUS", "BON,StMARTIN,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA", "GUA,BON,BAHAMAS,BERMUDE", "BON,StMARTIN,JAM,ARUBA,StEUS", "GUA,StMARTIN,StEUS,BERMUDE", "GUA,StMARTIN,StEUS,BAHAMAS", "GUA,StMARTIN,ARUBA,BERMUDE", "GUA,StMARTIN,ARUBA,BAHAMAS", "GUA,BON,StEUS", "GUA,BON,ARUBA", "BON,StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "BON,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,StEUS,BERMUDE", "GUA,BON,StEUS,BAHAMAS", "GUA,BON,ARUBA,BERMUDE", "GUA,BON,ARUBA,BAHAMAS", "BON,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS", "GUA,BON,StEUS,BAHAMAS,BERMUDE", "GUA,BON,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS,BAHAMAS", "GUA,BON,ARUBA,StEUS", "GUA,StMARTIN,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,ARUBA,StEUS,BERMUDE", "GUA,BON,ARUBA,StEUS,BAHAMAS", "GUA,BON,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM" };
        // the 39 missings
        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,StEUS", "GUA,ARUBA","StMARTIN,JAM,ARUBA,StEUS,BAHAMAS","BON,JAM,ARUBA,StEUS","GUA,ARUBA,BERMUDE","BON,JAM,ARUBA,StEUS,BERMUDE","BON,StMARTIN,BERMUDE","BON,StMARTIN,StEUS","GUA,ARUBA,StEUS,BERMUDE","BON,StMARTIN,StEUS,BERMUDE","BON,StMARTIN,ARUBA,BERMUDE","BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE","BON,StMARTIN,ARUBA,StEUS","BON,StMARTIN,ARUBA,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS,BAHAMAS","GUA,JAM,BAHAMAS","GUA,JAM,BAHAMAS,BERMUDE","GUA,JAM,StEUS","GUA,JAM,ARUBA,BERMUDE","GUA,JAM,ARUBA,BAHAMAS","BON,StMARTIN,JAM,BERMUDE","GUA,JAM,ARUBA,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,StEUS","GUA,JAM,ARUBA,StEUS,BAHAMAS","GUA,BON","GUA,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE","GUA,StMARTIN,StEUS,BERMUDE","GUA,StMARTIN,StEUS,BAHAMAS","GUA,StMARTIN,ARUBA,BAHAMAS","GUA,BON,StEUS","GUA,BON,ARUBA","BON,StMARTIN,JAM,ARUBA,StEUS,BERMUDE","GUA,StMARTIN,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,StEUS","GUA,BON,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,JAM" };
        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,ARUBA","StMARTIN,JAM,ARUBA,StEUS,BAHAMAS","BON,JAM,ARUBA,StEUS","BON,StMARTIN,BERMUDE","BON,StMARTIN,StEUS","BON,StMARTIN,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS","BON,StMARTIN,ARUBA,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS,BAHAMAS","GUA,JAM,BAHAMAS","GUA,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM","BON,StMARTIN,JAM,BERMUDE","BON,StMARTIN,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,StEUS","GUA,JAM,ARUBA,StEUS,BAHAMAS","GUA,BON","GUA,StMARTIN,StEUS,BERMUDE","GUA,StMARTIN,StEUS,BAHAMAS","GUA,StMARTIN,ARUBA,BAHAMAS","GUA,BON,StEUS","GUA,BON,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,StEUS", "GUA,StMARTIN,JAM,BERMUDE", "GUA,StMARTIN,JAM,BAHAMAS", "GUA,BON,JAM", "GUA,StMARTIN,JAM,BAHAMAS,BERMUDE", "GUA,BON,JAM,BERMUDE", "GUA,BON,JAM,BAHAMAS", "GUA,StMARTIN,JAM,StEUS", "GUA,StMARTIN,JAM,ARUBA", "GUA,BON,JAM,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,StEUS,BERMUDE", "GUA,StMARTIN,JAM,StEUS,BAHAMAS", "GUA,StMARTIN,JAM,ARUBA,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,BAHAMAS", "GUA,BON,JAM,StEUS", "GUA,BON,JAM,ARUBA", "GUA,StMARTIN,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,JAM,StEUS,BERMUDE", "GUA,BON,JAM,StEUS,BAHAMAS", "GUA,BON,JAM,ARUBA,BERMUDE", "GUA,BON,JAM,ARUBA,BAHAMAS", "GUA,StMARTIN,JAM,ARUBA,StEUS", "GUA,BON,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,BON,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "GUA,BON,JAM,ARUBA,StEUS", "GUA,BON,StMARTIN", "GUA,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,JAM,ARUBA,StEUS,BERMUDE", "GUA,BON,JAM,ARUBA,StEUS,BAHAMAS", "GUA,BON,StMARTIN,BERMUDE", "GUA,BON,StMARTIN,BAHAMAS", "GUA,BON,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,StEUS", "GUA,BON,StMARTIN,ARUBA", "GUA,BON,StMARTIN,StEUS,BERMUDE", "GUA,BON,StMARTIN,StEUS,BAHAMAS", "GUA,BON,StMARTIN,ARUBA,BERMUDE", "GUA,BON,StMARTIN,ARUBA,BAHAMAS", "GUA,BON,StMARTIN,StEUS,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,ARUBA,StEUS" };
        //searchSpace["MandatorySitesForTraining"] = "GUA,BON,ARUBA";
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_SMALL_128_401(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
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

            { nameof(NetworkSample.NumEpochs), 1},

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), false},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },


            { nameof(NetworkSample.AdamW_L2Regularization), 0.0005 },

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
            { nameof(NetworkSample.AlphaMixup), 0},
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
    private static void Launch_HPO_MEL_SPECTROGRAM_128_401_BinaryCrossentropy(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        //when LossFunction = BinaryCrossentropy
        //  InitialLearningRate ==
        //  AdamW_L2Regularization == 0.00025
        //  OneCycle_PercentInAnnealing == 0
        //  AlphaMixup == 1.2
        //  SkipConnectionsDropoutRate >= 0.4



        // stats from FindBestLearningRate
        //     DefaultMobileBlocksDescriptionCount     BatchSize   Best learning rate                           Free GPU Memory     
        //     B0 + -1                                 64          0.002 => 0.030 or 0.08                       12700MB             A305BB2D2F
        //     B0 + 5                                  64          0.003 => 0.025 or 0.06                       14587MB             93898B4B73  
        //     B0 + 5                                  128         ?     => 0.002                               7268MB              C05FD055A3
        //     B0 + -1                                 128         ?     => 0.01 or 0.1                         5500MB             A305BB2D2F                                       

        #region previous tests
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_128_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.93},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {128} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },

            
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.00015,  0.00025,  0.0005} },

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
            { nameof(NetworkSample.AlphaMixup), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0.1, 0.2} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),true/*new[]{true,false } */},

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };

        //model: FB0927A468_17
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = -1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.HorizontalFlip)] = true;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.0025;
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.94;
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;




        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = new[] { 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.02;
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy);
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[]{0,0.1};
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = new[]{5,-1}; //new[]{2,3,4,5,6, -1};
        //best: 0.9440      393C258A23
        //AdamW_L2Regularization = 0.00025
        //LossFunction = BinaryCrossentropy
        //DefaultMobileBlocksDescriptionCount = 5
        //OneCycle_PercentInAnnealing = 0



        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025; //new[] { 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.02;
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy);
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = new[] { 0.1f, 0.2f, 0.3f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0.1f, 0.2f, 0.3f, 0.4f };
        //best: 0.9514 89B2D5CEDF
        //AdamW_L2Regularization = 0.00025
        //DefaultMobileBlocksDescriptionCount = 5
        //LossFunction = BinaryCrossentropy
        //OneCycle_PercentInAnnealing = 0
        //TopDropoutRate = 0.2
        //SkipConnectionsDropoutRate = 0.2
        //TopDropoutRate = 0.1


        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy);
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = new[] { 0.0f, 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0.0f, 0.2f, 0.4f, 0.5f };
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.AlphaMixup)] = new[] { 1, 1.2, 1.5};
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        //best: 0.9511 9CEE8F0073
        //AlphaMixup = 1.2
        //LossFunction = BinaryCrossentropy
        //SkipConnectionsDropoutRate = 0.5
        //TopDropoutRate = 0.4
        //OneCycle_PercentInAnnealing = 0
        #region stats (hpo_16380.csv)
        /*
         * Stats for AlphaMixup:
         1.2:-0.9425161921459696 +/- 0.005026113550600352 (23 evals at 549.6s/eval) (target Time: 44.9%)
         1.5:-0.9396843284368515 +/- 0.00708116208620747 (20 evals at 546.7s/eval) (target Time: 30%)
         1:-0.9385012310484181 +/- 0.007097887046079005 (23 evals at 545.1s/eval) (target Time: 25.1%)
        Stats for BCEWithFocalLoss_Gamma:
         0:-0.9402588985183022 +/- 0.0066713671938491695 (66 evals at 547.1s/eval) (target Time: 50%)
         0.7:empty (target Time: 50%)
        Stats for BCEWithFocalLoss_PercentageInTrueClass:
         0.5:-0.9402588985183022 +/- 0.0066713671938491695 (66 evals at 547.1s/eval) (target Time: 33.3%)
         0.4:empty (target Time: 33.3%)
         0.6:empty (target Time: 33.3%)
        Stats for OneCycle_PercentInAnnealing:
         0:-0.9420196516173226 +/- 0.0062835303533355265 (35 evals at 551.3s/eval) (target Time: 64.3%)
         0.1:-0.9382709514710211 +/- 0.006536636545131606 (31 evals at 542.4s/eval) (target Time: 35.7%)
        Stats for SkipConnectionsDropoutRate:
         0.4:-0.9408757798373699 +/- 0.0073601562871463844 (16 evals at 546.1s/eval) (target Time: 26.9%)
         0.2:-0.9406338754822227 +/- 0.004946303973481242 (17 evals at 553s/eval) (target Time: 26.1%)
         0.5:-0.9405490942299366 +/- 0.006036560713137301 (16 evals at 544s/eval) (target Time: 25.8%)
         0:-0.939030201996074 +/- 0.0078074921220547475 (17 evals at 545.3s/eval) (target Time: 21.2%)
        Stats for TopDropoutRate:
         0:-0.941954721575198 +/- 0.005152757288117706 (23 evals at 551s/eval) (target Time: 42%)
         0.4:-0.9401502013206482 +/- 0.0074750716188162385 (22 evals at 545.6s/eval) (target Time: 33%)
         0.2:-0.938515441758292 +/- 0.006786811112259223 (21 evals at 544.5s/eval) (target Time: 25%)
         */
        #endregion




        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025; //new[] { 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.2; //1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[]{0.01,0.02,0.04};
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy);
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = 0.2f; //new[] { 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0f, 0.2f, 0.4f};
        //best 0.9502 7046B1D862
        //SkipConnectionsDropoutRate = 0
        //InitialLearningRate = 0.02
        //TopDropoutRate = 0.2
        #region stats (hpo_25492.csv)
        /*
        Stats for InitialLearningRate:
         0.02:-0.9446551203727722 +/- 0.003921505446990086 (3 evals at 559.1s/eval) (target Time: 55.7%)
         0.01:-0.9432544112205505 +/- 0.0018052004396709764 (3 evals at 563.8s/eval) (target Time: 38.8%)
         0.04:-0.9311263759930929 +/- 0.00738073029813322 (3 evals at 565.6s/eval) (target Time: 5.6%)
        Stats for SkipConnectionsDropoutRate:
         0:-0.9420586824417114 +/- 0.006738606448944757 (3 evals at 555s/eval) (target Time: 41.8%)
         0.4:-0.9406238198280334 +/- 0.0014258936491357123 (3 evals at 551.9s/eval) (target Time: 33.8%)
         0.2:-0.9363534053166708 +/- 0.010897142170991284 (3 evals at 581.6s/eval) (target Time: 24.5%)
         */
        #endregion
        #endregion

        //with Data Augmentation on waveform
        searchSpace = new Dictionary<string, object>
        {
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_128_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)},
            { nameof(NetworkSample.BatchSize), new[] {128} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },


            { nameof(NetworkSample.AdamW_L2Regularization), new[] { 0.000125, 0.00025, 0.0005 } },

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
            { nameof(NetworkSample.AlphaMixup), 1.2 },
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
        //AdamW_L2Regularization = 0.0005
        //CutoutPatchPercentage = 0.05
        //DefaultMobileBlocksDescriptionCount = 5
        //HeightShiftRangeInPercentage = 0.05
        //InitialLearningRate = 0.01
        //OneCycle_PercentInAnnealing = 0.1
        //RowsCutoutPatchPercentage = 0.1
        //SkipConnectionsDropoutRate = 0
        //TopDropoutRate = 0.5
        #region stats (hpo)
        /*
        Stats for AdamW_L2Regularization:
         0.00025:-0.9055530903515993 +/- 0.09953088082035416 (54 evals at 554.6s/eval) (target Time: 38.4%)
         0.0005:-0.877870343980335 +/- 0.1317073628888182 (21 evals at 568.6s/eval) (target Time: 31.1%)
         0.000125:-0.8759193297090202 +/- 0.1304431866656265 (29 evals at 575.6s/eval) (target Time: 30.5%)
        Stats for CutoutPatchPercentage:
         0:-0.9043973714877398 +/- 0.10099409896939227 (39 evals at 558s/eval) (target Time: 37.3%)
         0.05:-0.8874849336487907 +/- 0.1238971924906097 (42 evals at 566.3s/eval) (target Time: 32.5%)
         0.1:-0.8778669549071271 +/- 0.1251656872793232 (23 evals at 566.8s/eval) (target Time: 30.2%)
        Stats for DefaultMobileBlocksDescriptionCount:
         5:-0.9328659780820211 +/- 0.014218148134131598 (75 evals at 540.1s/eval) (target Time: 70.5%)
         -1:-0.7852364244132206 +/- 0.18035574752376546 (29 evals at 623.3s/eval) (target Time: 29.5%)
        Stats for HeightShiftRangeInPercentage:
         0:-0.8927666483254268 +/- 0.09257094835157732 (29 evals at 565s/eval) (target Time: 50.3%)
         0.05:-0.891287624835968 +/- 0.12467128503616376 (75 evals at 562.6s/eval) (target Time: 49.7%)
        Stats for InitialLearningRate:
         0.01:-0.9101193526695515 +/- 0.09675270521461657 (29 evals at 550.8s/eval) (target Time: 39.7%)
         0.005:-0.8933569073677063 +/- 0.11421729557163274 (60 evals at 572.4s/eval) (target Time: 34.2%)
         0.02:-0.8494619329770406 +/- 0.1468119232479954 (15 evals at 550.8s/eval) (target Time: 26.1%)
        Stats for OneCycle_PercentInAnnealing:
         0.1:-0.8936671334870008 +/- 0.11425247446878099 (49 evals at 555.7s/eval) (target Time: 50.8%)
         0:-0.8899475476958535 +/- 0.11865083929206394 (55 evals at 570.1s/eval) (target Time: 49.2%)
        Stats for RowsCutoutPatchPercentage:
         0.1:-0.8960317853671401 +/- 0.11730019684567332 (67 evals at 561s/eval) (target Time: 35.3%)
         0:-0.8936418169423154 +/- 0.0807621718442116 (19 evals at 563s/eval) (target Time: 34.6%)
         0.05:-0.8735266957018111 +/- 0.1416569056303118 (18 evals at 572s/eval) (target Time: 30.1%)
        Stats for SkipConnectionsDropoutRate:
         0:-0.9289139120475106 +/- 0.056270076240568116 (46 evals at 552.7s/eval) (target Time: 44.7%)
         0.2:-0.8624956828576548 +/- 0.14624622548564203 (27 evals at 571.4s/eval) (target Time: 28.2%)
         0.5:-0.8619155249288005 +/- 0.13641952621539116 (31 evals at 572s/eval) (target Time: 27.1%)
        Stats for TopDropoutRate:
         0.2:-0.9040996193885803 +/- 0.10182528176748463 (40 evals at 562.1s/eval) (target Time: 36.9%)
         0.5:-0.9016091050328435 +/- 0.10332706932548322 (37 evals at 555.4s/eval) (target Time: 36%)
         0:-0.8597512223102428 +/- 0.14493542054474506 (27 evals at 575.9s/eval) (target Time: 27.1%)
        */
        #endregion



        //TO find the best learning rate
        //searchSpace[nameof(NetworkSample.BatchSize)] = 8;
        //searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 1.0;

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_128_401_BCEWithFocalLoss(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        //when LossFunction = BCEWithFocalLoss && PercentageInTraining == 0.5
        //  InitialLearningRate == 0.01
        //  AdamW_L2Regularization == 0.0005
        //  AlphaMixup == 1.2

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

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.00015,  0.00025,  0.0005} },

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
            { nameof(NetworkSample.AlphaMixup), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0.1, 0.2} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),true/*new[]{true,false } */},

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };



        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = new[] { 0.000125, 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.2; //1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = new[] { 0, 0.7 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass)] = new[] { 0.4, 0.5, 0.6 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[] { 0.01, 0.02, 0.04 };
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss); //new[]{nameof(EvaluationMetricEnum.BinaryCrossentropy), nameof(EvaluationMetricEnum.BCEWithFocalLoss)};
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = new[] { 0.0f, 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0.0f, 0.2f, 0.4f, 0.5f };
        //best score: 0.9530 DEE753FA84
        //frequencies filtered in [1Khz, 127 KHz]
        //AdamW_L2Regularization = 0.0005
        //AlphaMixup = 1.2
        //BCEWithFocalLoss_Gamma = 0.7
        //InitialLearningRate = 0.01
        //LossFunction = BCEWithFocalLoss
        //OneCycle_PercentInAnnealing = 0.1
        //SkipConnectionsDropoutRate = 0.4
        //TopDropoutRate = 0.4
        #region stats hpo_14968.csv
        /*
        Stats for AdamW_L2Regularization:
         0.0005:-0.9475946337618726 +/- 0.003874275395588275 (47 evals at 559.3s/eval) (target Time: 57.2%)
         0.00025:-0.9441028104888068 +/- 0.004865525116561218 (18 evals at 551.8s/eval) (target Time: 27%)
         0.000125:-0.9395049611727396 +/- 0.007121081470705447 (12 evals at 551.1s/eval) (target Time: 15.8%)
        Stats for BCEWithFocalLoss_Gamma:
         0.7:-0.9463937282562256 +/- 0.004299672373975014 (55 evals at 557.8s/eval) (target Time: 60.2%)
         0:-0.9433274025266821 +/- 0.007532276866440651 (22 evals at 552.5s/eval) (target Time: 39.8%)
        Stats for BCEWithFocalLoss_PercentageInTrueClass:
         0.5:-0.9466376658529043 +/- 0.004914843386027181 (32 evals at 560.4s/eval) (target Time: 41%)
         0.6:-0.9458813454423632 +/- 0.004566989560571666 (28 evals at 555s/eval) (target Time: 35.2%)
         0.4:-0.9428102900000179 +/- 0.007224465977089863 (17 evals at 550.7s/eval) (target Time: 23.8%)
        Stats for InitialLearningRate:
         0.01:-0.9469701013867817 +/- 0.004018769666231349 (63 evals at 559.1s/eval) (target Time: 70.6%)
         0.02:-0.9405759990215301 +/- 0.005814162150319022 (10 evals at 546.2s/eval) (target Time: 20.5%)
         0.04:-0.9349953830242157 +/- 0.007700915063784506 (4 evals at 538.2s/eval) (target Time: 8.8%)
        Stats for OneCycle_PercentInAnnealing:
         0.1:-0.9458404429580854 +/- 0.004938108742290611 (46 evals at 558.4s/eval) (target Time: 53.1%)
         0:-0.9450386301163705 +/- 0.006423515574597289 (31 evals at 553.2s/eval) (target Time: 46.9%)
        Stats for SkipConnectionsDropoutRate:
         0:-0.9464393827048215 +/- 0.004338147617049178 (22 evals at 556.5s/eval) (target Time: 30.9%)
         0.2:-0.946089988663083 +/- 0.004247042477788084 (21 evals at 553.8s/eval) (target Time: 28.5%)
         0.4:-0.945903303367751 +/- 0.005022418648494283 (28 evals at 560s/eval) (target Time: 27.8%)
         0.5:-0.9383348723252615 +/- 0.00978371785892827 (6 evals at 546.9s/eval) (target Time: 12.8%)
        Stats for TopDropoutRate:
         0.2:-0.9457620173692703 +/- 0.004443007135792191 (20 evals at 553.3s/eval) (target Time: 34.5%)
         0:-0.9455617620394781 +/- 0.004844876735403736 (26 evals at 555.5s/eval) (target Time: 33.1%)
         0.4:-0.9453229596537929 +/- 0.006732685834161584 (31 evals at 558.9s/eval) (target Time: 32.3%)
        */
        #endregion

        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = new[] { 0.0005, 0.001 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = new[] { 1, 1.2, 1.5 }; //1.2; //1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = 0.7; //new[] { 0, 0.7 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass)] = 0.5; //new[] { 0.4, 0.5, 0.6 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[] { 0.001, 0.005, 0.01 };
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss); //new[]{nameof(EvaluationMetricEnum.BinaryCrossentropy), nameof(EvaluationMetricEnum.BCEWithFocalLoss)};
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0.1; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        //best score: 0.95264935 DEE753FA84_17 (hpo_21260.csv)
        //frequencies filtered in [1Khz, 127 KHz]
        //AdamW_L2Regularization = 0.0005
        //AlphaMixup = 1.2
        //InitialLearningRate = 0.01
        //LossFunction = BCEWithFocalLoss
        //PercentageInTraining = 0.5
        //SkipConnectionsDropoutRate = 0.4
        //TopDropoutRate = 0.4
        #region stats  (hpo_21260.csv)
        /*
        Stats for AdamW_L2Regularization:
         0.0005:-0.9448478031158447 +/- 0.0038517154463265633 (25 evals at 555.4s/eval) (target Time: 57.8%)
         0.001:-0.9433955659991816 +/- 0.004668487760529163 (38 evals at 547.2s/eval) (target Time: 42.2%)
        Stats for AlphaMixup:
         1.2:-0.9440867602825165 +/- 0.00512021404690902 (22 evals at 549.9s/eval) (target Time: 34.1%)
         1:-0.9440645830971854 +/- 0.004245862810450181 (21 evals at 553.4s/eval) (target Time: 34%)
         1.5:-0.9437480807304383 +/- 0.0037019231444786107 (20 evals at 548s/eval) (target Time: 31.9%)
        Stats for InitialLearningRate:
         0.01:-0.9458138346672058 +/- 0.0037814263715099873 (36 evals at 552.1s/eval) (target Time: 65.1%)
         0.005:-0.9428372634084601 +/- 0.0034752833199035464 (19 evals at 546.5s/eval) (target Time: 28.3%)
         0.001:-0.9383775666356087 +/- 0.0033481430877265778 (8 evals at 552.5s/eval) (target Time: 6.5%)
        Stats for SkipConnectionsDropoutRate:
         0.4:-0.9450428019399228 +/- 0.0038817164187930756 (23 evals at 551.1s/eval) (target Time: 41.9%)
         0.2:-0.94401577824638 +/- 0.004265958047158649 (21 evals at 548.9s/eval) (target Time: 32.9%)
         0.1:-0.9426268841090956 +/- 0.004819821059514996 (19 evals at 551.4s/eval) (target Time: 25.1%)
        Stats for TopDropoutRate:
         0.2:-0.9445303471192069 +/- 0.004591580714187518 (23 evals at 550.1s/eval) (target Time: 37.6%)
         0.1:-0.9438837484309548 +/- 0.0037630135111553814 (19 evals at 553.5s/eval) (target Time: 32.6%)
         0.4:-0.9434398753302438 +/- 0.004698970635829483 (21 evals at 548.1s/eval) (target Time: 29.8%)
         */
        #endregion


        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.0005; //new[] { 0.0005, 0.001 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.2; //new[] { 1, 1.2, 1.5 }; //1.2; //1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = 0.7; //new[] { 0, 0.7 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass)] = 0.5; //new[] { 0.4, 0.5, 0.6 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.01; //new[] { 0.001, 0.005, 0.01 };
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss); //new[]{nameof(EvaluationMetricEnum.BinaryCrossentropy), nameof(EvaluationMetricEnum.BCEWithFocalLoss)};
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0.1; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        //best score 0.9520344 9F981742E0
        //using no min max for frequencies (instead of [1Khz, 127Khz]
        //HorizontalFlip = False
        //LossFunction = BCEWithFocalLoss
        //PercentageInTraining = 0.5
        //SkipConnectionsDropoutRate = 0.1
        //TopDropoutRate = 0.4
        #region stats  (hpo_2838800.csv)
        /*
        Stats for HorizontalFlip:
         False:-0.9489981532096863 +/- 0.0027299571800770924 (8 evals at 575.6s/eval) (target Time: 62.7%)
         True:-0.9476127113614764 +/- 0.002666123989124074 (7 evals at 563.1s/eval) (target Time: 37.3%)
        Stats for SkipConnectionsDropoutRate:
         0.4:-0.9488059878349304 +/- 0.002439593718939452 (6 evals at 564.6s/eval) (target Time: 39.2%)
         0.2:-0.9487205843130747 +/- 0.0019134571086830776 (6 evals at 576.8s/eval) (target Time: 37.8%)
         0.1:-0.9467049241065979 +/- 0.004028103288209574 (3 evals at 566s/eval) (target Time: 23%)
        Stats for TopDropoutRate:
         0.2:-0.9499077647924423 +/- 0.0010207306463129444 (4 evals at 587.3s/eval) (target Time: 51.6%)
         0.4:-0.9491305450598398 +/- 0.002009837369652986 (6 evals at 565.5s/eval) (target Time: 34.9%)
         0.1:-0.9461719751358032 +/- 0.003195609504752072 (5 evals at 560.9s/eval) (target Time: 13.5%)
         */
        #endregion

        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.0005; //new[] { 0.0005, 0.001 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.2; //new[] { 1, 1.2, 1.5 }; //1.2; //1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = 0.7; //new[] { 0, 0.7 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass)] = 0.5; //new[] { 0.4, 0.5, 0.6 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[] { 0.005, 0.01 };
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss); //new[]{nameof(EvaluationMetricEnum.BinaryCrossentropy), nameof(EvaluationMetricEnum.BCEWithFocalLoss)};
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[] { 0, 0.1 }; //0.1; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = new[] { 0.2f, 0.4f };  //0.2f; //new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.8;
        //best score 0.9362 346AA4C20A
        //frequencies filtered in [1Khz, 127 KHz]
        //InitialLearningRate = 0.01
        //LossFunction = BCEWithFocalLoss
        //OneCycle_PercentInAnnealing = 0
        //PercentageInTraining = 0.8
        //SkipConnectionsDropoutRate = 0.1
        //TopDropoutRate = 0.4
        #region stats (hpo_27500.csv)
        /*
        Stats for InitialLearningRate:
         0.01:-0.9296465267737707 +/- 0.004080871365891207 (12 evals at 638s/eval) (target Time: 54.7%)
         0.005:-0.9285298188527426 +/- 0.005996010820847613 (12 evals at 633.9s/eval) (target Time: 45.3%)
        Stats for OneCycle_PercentInAnnealing:
         0:-0.9319609055916468 +/- 0.004194974567202304 (12 evals at 637s/eval) (target Time: 82.8%)
         0.1:-0.9262154400348663 +/- 0.004373348409127221 (12 evals at 634.8s/eval) (target Time: 17.2%)
        Stats for SkipConnectionsDropoutRate:
         0.1:-0.9326568320393562 +/- 0.002140197900684004 (8 evals at 642.5s/eval) (target Time: 60.9%)
         0.4:-0.9289667829871178 +/- 0.005012753511225588 (8 evals at 637.7s/eval) (target Time: 28.1%)
         0.2:-0.9256409034132957 +/- 0.005049939350394046 (8 evals at 627.6s/eval) (target Time: 11%)
        Stats for TopDropoutRate:
         0.4:-0.9300025949875513 +/- 0.004669493688728327 (12 evals at 632.7s/eval) (target Time: 58.4%)
         0.2:-0.9281737506389618 +/- 0.005454600629231032 (12 evals at 639.2s/eval) (target Time: 41.6%)
        */
        #endregion






        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.0005;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.2; //new[] { 1, 1.2, 1.5 }; //1.2; //1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = 0.7; //new[] { 0, 0.7 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass)] = 0.5; //new[] { 0.4, 0.5, 0.6 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.01; //new[] { 0.001, 0.005, 0.01 };
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss); //new[]{nameof(EvaluationMetricEnum.BinaryCrossentropy), nameof(EvaluationMetricEnum.BCEWithFocalLoss)};
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0.1; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = 0.4f; // new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = 0.4f; //new[] { 0.1f, 0.2f, 0.4f };
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        //with random length for RowsCutoutPatchPercentage && CutoutPatchPercentage
        //best score: 0.95264935 DEE753FA84_17 (hpo_21260.csv)
        //frequencies filtered in [1Khz, 127 KHz]
        //AdamW_L2Regularization = 0.0005
        //AlphaMixup = 1.2
        //InitialLearningRate = 0.01
        //LossFunction = BCEWithFocalLoss
        //PercentageInTraining = 0.5
        //SkipConnectionsDropoutRate = 0.4
        //TopDropoutRate = 0.4



        //To find the best learning rate
        //searchSpace[nameof(NetworkSample.BatchSize)] = 8;
        //searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 1.0;

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_256_801_BinaryCrossentropy(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        //for numEpochs = 20
        //HorizontalFlip = False
        //AdamW_L2Regularization: 0.000125 (or 0.0000625)
        //InitialLearningRate: 0.0025, (or 0.00125)
        // stats from FindBestLearningRate
        //     DefaultMobileBlocksDescriptionCount     BatchSize   Best learning rate                           Free GPU Memory
        //     B0 + -1                                 8           0.0001   0.001
        //     B0 + -1                                 32          0.006    0.02 (between 0.01 and 0.025)
        //     B2 + -1                                 8           ?        0.045                               16000 MB/25GB   D58CB83204
        //     B2 + -1                                 16          ?        0.05                                10171 MB/25GB   32093F4126
        //     B2 + -1                                 32          ?        0.05                                  666 MB/25GB   9D677FD756

        #region previous tests
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
            { nameof(NetworkSample.BatchSize), new[] {32} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.00015,  0.00025,  0.0005} },
            
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
            { nameof(NetworkSample.AlphaMixup), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0.1, 0.2} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            
            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},
        };
        #endregion

        searchSpace = new Dictionary<string, object>
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

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization),  new[]{0.0000625,0.000125,0.00025}},

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
            { nameof(NetworkSample.AlphaMixup), 1.2 },
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
        //AdamW_L2Regularization = 0.000125
        //HorizontalFlip = False
        //InitialLearningRate = 0.0025
        //for: 07D3470DB7_16, Valid Score: 0.9546, Test score AUC: 0.938326148145509
        #region stats (hpo_18276.csv)
        /*
        Stats for AdamW_L2Regularization:
         6.25E-05:-0.9526493748029073 +/- 0.0012039836999502 (6 evals at 2462.1s/eval) (target Time: 39.4%)
         0.000125:-0.9524956345558167 +/- 0.002981476658451587 (6 evals at 2468s/eval) (target Time: 37.4%)
         0.00025:-0.9503535866737366 +/- 0.004427104142241886 (5 evals at 2508s/eval) (target Time: 23.2%)
        Stats for HorizontalFlip:
         False:-0.9532643109560013 +/- 0.001495791456737716 (8 evals at 2448.9s/eval) (target Time: 66.6%)
         True:-0.9507248335414462 +/- 0.003833549650885314 (9 evals at 2503.2s/eval) (target Time: 33.4%)
        Stats for InitialLearningRate:
         0.00125:-0.9534692962964376 +/- 0.0013017994292096654 (6 evals at 2467.5s/eval) (target Time: 47.6%)
         0.0025:-0.9529773354530334 +/- 0.001915661825006404 (5 evals at 2602.6s/eval) (target Time: 36.7%)
         0.005:-0.9494892557462057 +/- 0.003947943378510277 (6 evals at 2383.6s/eval) (target Time: 15.7%)
        */
        #endregion


        //TO find the best learning rate
        //searchSpace[nameof(NetworkSample.BatchSize)] = 8;
        //searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 1.0;

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_256_801_BCEWithFocalLoss(int numEpochs = 20, int maxAllowedSecondsForAllComputation = 0)
    {
        // DefaultMobileBlocksDescriptionCount = 6 (bad: 5)
        // PercentageInTraining = 0.5
        // AlphaRowsCutMix = 0 (bad: 1)

        #region previous tests
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_256_801)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), 0.7},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            { nameof(NetworkSample.BatchSize), new[] {32} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] { 0.000125, 0.00025, 0.0005 }},

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new[] { 0.005, 0.01, 0.02 } },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[] { 0, 0.1 }},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            //{nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},

            {nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate), 0.4},
            {nameof(EfficientNetNetworkSample.TopDropoutRate), 0.4},

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), 0.1477559 /*new[] {0, 0.1, 0.2}*/ },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.12661687 /*new[] {0.1, 0.2}*/ },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            { nameof(NetworkSample.HorizontalFlip), new[]{true,false } },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), 0.06299627 /*new[]{0, 0.05}*/ },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };
        //best score 0.951522 (F8FBA26AB7)
        //AdamW_L2Regularization = 0.000125
        //HorizontalFlip = False
        //InitialLearningRate = 0.005
        //OneCycle_PercentInAnnealing = 0
        #region stats (hpo_28692.csv)
        /*
        Stats for AdamW_L2Regularization:
         0.000125:-0.9395733028650284 +/- 0.00686067682360214 (12 evals at 1246.8s/eval) (target Time: 39.5%)
         0.00025:-0.9387533714373907 +/- 0.00538090237567531 (12 evals at 1234s/eval) (target Time: 35%)
         0.0005:-0.9363448570171992 +/- 0.007498458417104663 (12 evals at 1242.2s/eval) (target Time: 25.5%)
        Stats for HorizontalFlip:
         False:-0.9395590689447191 +/- 0.00621623967913609 (18 evals at 1251.8s/eval) (target Time: 59.5%)
         True:-0.936888618601693 +/- 0.007050814312582564 (18 evals at 1230.2s/eval) (target Time: 40.5%)
        Stats for InitialLearningRate:
         0.005:-0.9427675952514013 +/- 0.007217818169133016 (12 evals at 1273.6s/eval) (target Time: 56.5%)
         0.01:-0.9388985683520635 +/- 0.004694809306424899 (12 evals at 1247.8s/eval) (target Time: 32.6%)
         0.02:-0.9330053677161535 +/- 0.003925601794271406 (12 evals at 1201.5s/eval) (target Time: 10.9%)
        Stats for OneCycle_PercentInAnnealing:
         0.1:-0.9393996364540524 +/- 0.005619643922845447 (18 evals at 1239.3s/eval) (target Time: 57.7%)
         0:-0.9370480510923598 +/- 0.0075878782317587885 (18 evals at 1242.7s/eval) (target Time: 42.3%)
        */
        #endregion

        searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_256_801)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), 0.7},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            { nameof(NetworkSample.BatchSize), new[] {32} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] { 0.000125, 0.00025}},

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), new[]{6,-1} },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new[] { 0.00125, 0.0025, 0.005 } },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[] { 0, 0.1 }},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            //{nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            //{nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},

            {nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate), 0.4},
            {nameof(EfficientNetNetworkSample.TopDropoutRate), 0.4},

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), 0.1477559 /*new[] {0, 0.1, 0.2}*/ },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.12661687 /*new[] {0.1, 0.2}*/ },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            //{ nameof(NetworkSample.HorizontalFlip), false /*new[]{true,false }*/ },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), 0.06299627 /*new[]{0, 0.05}*/ },
            //{ nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };
        //best results 0.95316184  (6F17982D0A) , test AUC: 0.951722663626322
        //AdamW_L2Regularization = 0.000125
        //DefaultMobileBlocksDescriptionCount = -1
        //InitialLearningRate = 0.0025
        //OneCycle_PercentInAnnealing = 0.1
        //SkipConnectionsDropoutRate = 0.4
        //TopDropoutRate = 0.4
        #region stats (hpo_23384.csv)
        /*
        Stats for AdamW_L2Regularization:
         0.000125:-0.9472686275839806 +/- 0.00351058497435207 (8 evals at 2545.2s/eval) (target Time: 68.6%)
         0.00025:-0.944655105471611 +/- 0.0017677829989537954 (4 evals at 2458.9s/eval) (target Time: 31.4%)
        Stats for DefaultMobileBlocksDescriptionCount:
         -1:-0.9464828670024872 +/- 0.003048080340943342 (6 evals at 2495.8s/eval) (target Time: 51.2%)
         6:-0.9463120400905609 +/- 0.003499434111791521 (6 evals at 2537.1s/eval) (target Time: 48.8%)
        Stats for InitialLearningRate:
         0.0025:-0.9492056965827942 +/- 0.0029007571975109745 (5 evals at 2588.3s/eval) (target Time: 82.1%)
         0.00125:-0.9446294903755188 +/- 0.0016227589301145721 (4 evals at 2472.4s/eval) (target Time: 9.7%)
         0.005:-0.9440743327140808 +/- 0.0016910158559540903 (3 evals at 2455.5s/eval) (target Time: 8.2%)
        Stats for OneCycle_PercentInAnnealing:
         0.1:-0.9485241323709488 +/- 0.003980929326076737 (4 evals at 2595.2s/eval) (target Time: 70%)
         0:-0.9453341141343117 +/- 0.0022017258167526346 (8 evals at 2477.1s/eval) (target Time: 30%)
        */
        #endregion

        searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled
            
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_256_801)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), new[]{0.7, 1.5}},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            { nameof(NetworkSample.BatchSize), new[] {42} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization),  0.000125},

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 6 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), 0.0025},

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
            {nameof(EfficientNetNetworkSample.TopDropoutRate), new[]{0.2, 0.4}},

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0.1, 0.2} /*0.1477559 */ },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0.1, 0.15} /*0.12661687 */ },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            //{ nameof(NetworkSample.HorizontalFlip), false /*new[]{true,false }*/ },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0.05, 0.1} /*0.06299627*/ },
            //{ nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };
        //best results 0.9542892 (9811DDD19E) , test AUC: 0.952791534542345
        //BCEWithFocalLoss_Gamma = 0.7
        //CutoutPatchPercentage = 0.2
        //HeightShiftRangeInPercentage = 0.05
        //RowsCutoutPatchPercentage = 0.15
        //SkipConnectionsDropoutRate = 0.4
        //TopDropoutRate = 0.2
        #region stats (hpo_3052.csv)
        /*
        Stats for BCEWithFocalLoss_Gamma:
         0.7:-0.9509815682064403 +/- 0.0027598442248482004 (11 evals at 2499.4s/eval) (target Time: 69.2%)
         1.5:-0.9480828642845154 +/- 0.0037676355471132445 (9 evals at 2402.3s/eval) (target Time: 30.8%)
        Stats for CutoutPatchPercentage:
         0.2:-0.9505141576131185 +/- 0.002998847351561212 (12 evals at 2479.5s/eval) (target Time: 63.3%)
         0.1:-0.9484216421842575 +/- 0.003940141698979831 (8 evals at 2420s/eval) (target Time: 36.7%)
        Stats for HeightShiftRangeInPercentage:
         0.05:-0.950268438229194 +/- 0.003297668726573796 (13 evals at 2474.8s/eval) (target Time: 61.2%)
         0.1:-0.9485790474074227 +/- 0.003756445936684401 (7 evals at 2420.2s/eval) (target Time: 38.8%)
        Stats for RowsCutoutPatchPercentage:
         0.15:-0.9498206317424774 +/- 0.003533527077467958 (10 evals at 2467.6s/eval) (target Time: 52%)
         0.1:-0.9495336711406708 +/- 0.0035757834113661928 (10 evals at 2443.8s/eval) (target Time: 48%)
        Stats for TopDropoutRate:
         0.2:-0.9497591435909272 +/- 0.00423641648383309 (10 evals at 2423.7s/eval) (target Time: 51%)
         0.4:-0.949595159292221 +/- 0.0027115574410394123 (10 evals at 2487.7s/eval) (target Time: 49%)*
        */
        #endregion

        searchSpace = new Dictionary<string, object>
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
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), 0.7},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            { nameof(NetworkSample.BatchSize), new[] {42} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization),  0.000125},

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 6 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), 0.0025},

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

            {nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate),new[]{0, 0.2, 0.4, 0.6}},
            {nameof(EfficientNetNetworkSample.TopDropoutRate), new[]{0, 0.2, 0.4, 0.6}},

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), 0.1477559 },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.12661687  },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            //{ nameof(NetworkSample.HorizontalFlip), false /*new[]{true,false }*/ },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), 0.06299627 },

        };
        //best results 0.95828635 (5373FD5B44) , test AUC: 0.947607510599636
        //SkipConnectionsDropoutRate = 0.4
        //TopDropoutRate = 0
        #region stats (hpo_11480.csv)
        /*
        Stats for SkipConnectionsDropoutRate:
         0.4:-0.9545710682868958 +/- 0.002953050433433706 (4 evals at 2406.4s/eval) (target Time: 32.7%)
         0:-0.9536742866039276 +/- 0.002315695194055391 (4 evals at 2472.7s/eval) (target Time: 24.1%)
         0.2:-0.9535461664199829 +/- 0.001813775627589635 (4 evals at 2389s/eval) (target Time: 23.1%)
         0.6:-0.95316182076931 +/- 0.001776674562473303 (4 evals at 2369.3s/eval) (target Time: 20.1%)
        Stats for TopDropoutRate:
         0.6:-0.9546991735696793 +/- 0.0008905567798440595 (4 evals at 2393.3s/eval) (target Time: 35.9%)
         0:-0.954186737537384 +/- 0.0027615352713985693 (4 evals at 2471.4s/eval) (target Time: 29.8%)
         0.4:-0.9536230266094208 +/- 0.0019479943720284924 (4 evals at 2384.1s/eval) (target Time: 20.4%)
         0.2:-0.9524444043636322 +/- 0.0025632930678885484 (4 evals at 2388.7s/eval) (target Time: 14%)       
        */
        #endregion

        searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), new[]{0.5,0.8}}, //will be automatically set to 1 if KFold is enabled
            
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_256_801)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.95},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), new[]{0.35, 0.7}},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            { nameof(NetworkSample.BatchSize), new[] {42} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), 0.0005 },
            { nameof(NetworkSample.AdamW_L2Regularization),  0.000125},

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount),new[]{5,6}},

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate),new[]{0.002, 0.0025}},

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

            {nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate),new[]{0.4, 0.75}},
            {nameof(EfficientNetNetworkSample.TopDropoutRate), 0.4},

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), 0.1477559 },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.12661687  },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            //{ nameof(NetworkSample.HorizontalFlip), false /*new[]{true,false }*/ },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), 0.06299627 },

        };
        //best results 0.9540842 (AB8803D9AA)
        //InitialLearningRate = 0.002
        //BCEWithFocalLoss_Gamma = 0.7
        //SkipConnectionsDropoutRate = 0.4
        //PercentageInTraining = 0.5
        //TopDropoutRate = 0.4
        #region stats (hpo_8448.csv)
        /*
        Stats for BCEWithFocalLoss_Gamma:
         0.35:-0.9434492588043213 +/- 0.006478073273873414 (4 evals at 2302.3s/eval) (target Time: 58.3%)
         0.7:-0.9388564705848694 +/- 0.013882268756469385 (5 evals at 2450.7s/eval) (target Time: 41.7%)
        Stats for DefaultMobileBlocksDescriptionCount:
         6:-0.9463829278945923 +/- 0.009372360606343106 (5 evals at 2545.6s/eval) (target Time: 80.8%)
         5:-0.9340411871671677 +/- 0.010007898687944283 (4 evals at 2183.7s/eval) (target Time: 19.2%)
        Stats for InitialLearningRate:
         0.0025:-0.9434572656949362 +/- 0.0074802175557260415 (3 evals at 2399.3s/eval) (target Time: 57.5%)
         0.002:-0.939617931842804 +/- 0.012786592108888037 (6 evals at 2377.5s/eval) (target Time: 42.5%)
        Stats for PercentageInTraining:
         0.5:-0.946790337562561 +/- 0.007447656238284497 (6 evals at 2222.3s/eval) (target Time: 90.9%)
         0.8:-0.929112454255422 +/- 0.008573271299997805 (3 evals at 2709.7s/eval) (target Time: 9.1%)
        Stats for SkipConnectionsDropoutRate:
         0.75:-0.9409775535265604 +/- 0.009165780088122386 (3 evals at 2364.8s/eval) (target Time: 50.2%)
         0.4:-0.9408577879269918 +/- 0.012424842708492695 (6 evals at 2394.8s/eval) (target Time: 49.8%)
        */
        #endregion

        searchSpace = new Dictionary<string, object>
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
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), 0.7},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            { nameof(NetworkSample.BatchSize), new[] {42} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), 0.0005 },
            { nameof(NetworkSample.AdamW_L2Regularization),  0.000125},

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount),new[]{5,6}},

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate),0.0025},

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
            { nameof(NetworkSample.AlphaMixup), 1.2 },
            { nameof(NetworkSample.CutoutPatchPercentage), 0.1477559 },
            { nameof(NetworkSample.CutoutCount), 1 },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.12661687  },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            //{ nameof(NetworkSample.ColumnsCutoutCount),  0 },
            //{ nameof(NetworkSample.HorizontalFlip), false /*new[]{true,false }*/ },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), 0.06299627 },

        };
        //best results 
        #region stats (hpo_8448.csv)
        /*
        Stats for BCEWithFocalLoss_Gamma:
        */
        #endregion
        #endregion //previous tests


        searchSpace = new Dictionary<string, object>
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

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), 0.0005 },
            { nameof(NetworkSample.AdamW_L2Regularization),  0.000125},

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
            { nameof(NetworkSample.AlphaMixup), 1.2 },
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
        //best results 
        #region stats (hpo_.csv)
        /*
        */
        #endregion

        //like 1EF57E45FC_16
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = 0.35;
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = 0.3;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.12661687;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.06299627;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.25;

        //changes on 1EF57E45FC_16
        searchSpace[nameof(NetworkSample.NumEpochs)] = new[]{/*50,100*/ 50};
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[] { /*0.001, */0.001};
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = 0.75; //new[] { 0.4, 0.5, 0.6, 0.65}; //0.4
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = 0.5; //new[] { 0.3, 0.5, 0.6, 0.65 }; //0.3



        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }




    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_64_401(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // stats from FindBestLearningRate
        //     DefaultMobileBlocksDescriptionCount     BatchSize   Best learning rate:
        //     B0 + -1                                 8           0.0001   0.015
        //     B0 + -1                                 32          0.001    0.025
        //     B0 + -1                                 64          0.0005   0.03
        //     B0 + -1                                 128         0.002    0.05

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

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.00015,  0.00025,  0.0005} },
            
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
            { nameof(NetworkSample.AlphaMixup), new[] { 1.0, 1.2} },
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
            lambdaL2Regularization = 0.0005,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
            NumEpochs = 10,
            BatchSize = 64,
            InitialLearningRate = 0.01,


            //Data augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            //AlphaMixup = 0.0,
            //AlphaCutMix = 0.0,
            //CutoutPatchPercentage = 0.0
        }
            .WithSGD(0.9, false)
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