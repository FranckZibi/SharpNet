using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using log4net;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using static SharpNet.Networks.NetworkSample;
using static SharpNet.Optimizers.Optimizer;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Networks.Transformers;

public static class MyNameIsGrootUtils
{
    public const string NAME = "MyNameIsGroot";


    #region public fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(MyNameIsGrootUtils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);

    public static void Run()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "log");

        //CharLevelTransformerInference();
        //Retrain();
        LaunchNeuralNetworkHPO(5);
        //SpeedTest();
        //PredictNextWord("97B423C404", "my name is", 1, 0.0);
        //PredictNextWord("97B423C404", "groot is my", 50, 0.0);
    }

    public static void Retrain()
    {
        ChallengeTools.Retrain(WorkingDirectory, "1E68E0FFA0", null, 0.9, retrainOnFullDataset: false, useAllAvailableCores: true);
    }

    public static void LaunchNeuralNetworkHPO(int num_epochs, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {

            //related to Dataset
            //{ "KFold", 3 },
            {nameof(AbstractDatasetSample.PercentageInTraining), 0.8},
            {nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},

            //related to model
            //{nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.SparseCategoricalCrossentropy)},
            //{nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.SparseAccuracy)},
            {nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.CategoricalCrossentropy)},
            {nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)},

            {nameof(NetworkSample.CompatibilityMode), new[]{CompatibilityModeEnum.PyTorch}},
            //{nameof(NetworkSample.CompatibilityMode), new[]{nameof(CompatibilityModeEnum.TensorFlow),nameof(CompatibilityModeEnum.PyTorch)}},

            {nameof(TransformerDatasetSample.max_length), 3 },
            {nameof(TransformerDatasetSample.vocab_size), 4},
            //{nameof(TransformerDatasetSample.layer_norm_epsilon), new[]{1e-5, 1e-6 } },
            {nameof(TransformerNetworkSample.embedding_dim), 2},
            {nameof(TransformerNetworkSample.decoder_num_transformer_blocks), new[]{1 /*,6*/ } },
            {nameof(TransformerNetworkSample.decoder_num_heads), new[]{1} },
            {nameof(TransformerNetworkSample.decoder_mha_use_bias_Q_K_V), false},
            {nameof(TransformerNetworkSample.decoder_mha_use_bias_O), false },
            //{nameof(TransformerNetworkSample.decoder_mha_dropout), new[]{0.2f,0f ,0.1f} },
            //{nameof(TransformerNetworkSample.decoder_add_layer_norm_before_mha), false},
            {nameof(TransformerNetworkSample.decoder_feed_forward_dim), 1},
            //{nameof(TransformerNetworkSample.decoder_feed_forward_dropout), new[]{0.2f,0f,0.1f }},
            {nameof(TransformerNetworkSample.flattenInputTensorOnLastDimension_in_last_dense), false},

            {nameof(TransformerNetworkSample.decoder_mha_dropout), new[]{0.0,0.2,0.4 } },
            {nameof(TransformerNetworkSample.decoder_feed_forward_dropout), new[]{0.0,0.2,0.4 } },


            

            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[]{OptimizationEnum.AdamW}},
            //{ nameof(NetworkSample.OptimizerType), new[]{nameof(OptimizationEnum.AdamW), nameof(OptimizationEnum.Adam), nameof(OptimizationEnum.SGD), nameof(OptimizationEnum.VanillaSGD)}},
            //{ nameof(NetworkSample.AdamW_L2Regularization), 0.01},
            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate), new[]{0.05 } },
            //{ nameof(NetworkSample.InitialLearningRate), new[]{10, 5, 1.0,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005, 0.00001, 0.000005, 0.000001 , 0.0000005, 0.0000001 } },
            { nameof(NetworkSample.InitialLearningRate), new[]{0.1, 0.5, 0.05 } },
            // Learning Rate Scheduler
            { nameof(NetworkSample.LearningRateSchedulerType), new[] { LearningRateSchedulerEnum.OneCycle } },
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "CyclicCosineAnnealing", "OneCycle", "Linear", "Constant" } },


            { nameof(NetworkSample.BatchSize), new[]{256} },

            { nameof(NetworkSample.num_epochs), num_epochs },

            //{ "ResourceIds", -1},
            


        };

        var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(GetModelSample(), new MyNameIsGrootDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, false, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    private static string GenerateText(Network nn, int textLength, double maxAllowedError)
    {
        var outputShape = nn.YPredicted_MiniBatch_Shape(1);
        var max_length = outputShape[1];
        var vocab_size = outputShape[2];
        var datasetSample = new MyNameIsGrootDatasetSample() { max_length = max_length, vocab_size = vocab_size };
        var tokenizer = datasetSample.GetTokenizer();

        var xInputSingleRow = new CpuTensor<float>(new[] { 1, max_length});
        var xInputSingleRowSpan = xInputSingleRow.SpanContent;

        var fulltext = datasetSample.GetText();
        var r = new Random();
        int randomStartIdx = r.Next(fulltext.Length/2 - max_length-1);

        List<int> tmpSequence = tokenizer.TextsToSequences(new[] { fulltext.Substring(randomStartIdx, 10*max_length) })[0].Take(max_length).ToList();

        int[] newSequence = new int[max_length+textLength];
        for (int j = 0; j < max_length; j++)
        {
            newSequence[j] = tmpSequence[j];
        }

        int nextIndexToGenerate = max_length;
        while (nextIndexToGenerate < newSequence.Length)
        {
            //we know the sequence newSequence[nextIndexToGenerate-max_length] to newSequence[nextIndexToGenerate-1]
            //we want to compute next sequence item at position newSequence[nextIndexToGenerate]

            for (int i = 0; i < max_length; i++)
            {
                xInputSingleRowSpan[i] = newSequence[nextIndexToGenerate - max_length+i];
            }

            var prediction = nn.Predict(xInputSingleRow, false);
            var proba = prediction.As2DTensor(true).RowSlice(max_length-1, 1).ContentAsFloatArray();
            var indexNextToken = GetIndexPrediction(proba, r, maxAllowedError);
            newSequence[nextIndexToGenerate] = indexNextToken;
            ++nextIndexToGenerate;
        }
        var generateText  =tokenizer.SequenceToText(newSequence.Skip(max_length));
        return generateText;
    }


    private static int GetIndexPrediction(float[] proba, Random r, double maxAllowedError)
    {
        List<Tuple<float, int>> probaWithIndex = new List<Tuple<float, int>>();
        for (int i = 0; i < proba.Length; i++)
        {
            probaWithIndex.Add(new Tuple<float, int>(proba[i], i));
        }
        probaWithIndex = probaWithIndex.OrderByDescending(x => x.Item1).ToList();
        int selectionChoice = 1;
        for (int i = 1; i < probaWithIndex.Count; i++)
        {
            if (probaWithIndex[i].Item1 > (1.0*probaWithIndex[0].Item1- maxAllowedError))
            {
                ++selectionChoice;
            }
            else
            {
                break;
            }
        }

        return probaWithIndex[r.Next(selectionChoice)].Item2;
    }


    private static string PredictNextWord(string networkName, string startingText, int nbTokenToPredictAfterText,double maxAllowedError)
    {
        using var nn = Network.LoadTrainedNetworkModel(WorkingDirectory, networkName);
        nn.Sample.LogNetworkPropagation = true;
        //nn.Sample.SetResourceId(-1);
        var datasetSample = (MyNameIsGrootDatasetSample)ModelAndDatasetPredictionsSample.LoadDatasetSample(WorkingDirectory, networkName);
        var max_length = datasetSample.max_length;
        var tokenizer = datasetSample.GetTokenizer();

        var xInputSingleRow = new CpuTensor<float>(new[] { 1, max_length });
        var xInputSingleRowSpan = xInputSingleRow.SpanContent;

        Log.Info($"Starting text = {startingText}");
        var r = new Random();

        List<int> startingTextSequence = tokenizer.TextsToSequences(new[] { startingText })[0].ToList();

        int[] newSequence = new int[max_length + nbTokenToPredictAfterText];
        int idxFromStartingText = startingTextSequence.Count-1;
        for (int j = max_length-1; j >=0; --j)
        {
            newSequence[j] = (idxFromStartingText>=0)?startingTextSequence[idxFromStartingText--] :0;
        }

        int nextIndexToGenerate = max_length;
        while (nextIndexToGenerate < newSequence.Length)
        {
            //we know the sequence newSequence[nextIndexToGenerate-max_length] to newSequence[nextIndexToGenerate-1]
            //we want to compute next sequence item at position newSequence[nextIndexToGenerate]

            for (int i = 0; i < max_length; i++)
            {
                xInputSingleRowSpan[i] = newSequence[nextIndexToGenerate - max_length + i];
            }

            var prediction = nn.Predict(xInputSingleRow, false);
            var proba = prediction.ContentAsFloatArray();
            var indexNextToken = GetIndexPrediction(proba, r, maxAllowedError);
            newSequence[nextIndexToGenerate] = indexNextToken;
            ++nextIndexToGenerate;
        }
        var generateText = tokenizer.SequenceToText(newSequence.Skip(max_length));
        Log.Info($"Generated text = {generateText}");
        return generateText;
    }

    public static void CharLevelTransformerInference()
    {
        var nn = Network.LoadTrainedNetworkModel(WorkingDirectory, "340065842F");
        foreach (var maxAllowedError in new[] { 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16 })
        {
            Log.Info($"---------------------------");
            Log.Info($"maxAllowedError={maxAllowedError}");
            var textGenerated = GenerateText(nn, 2000, maxAllowedError);
            Log.Info(textGenerated);
        }
        return;
    }

    private static void SpeedTest()
    {
        var speedTestWorkingDirectory = Path.Join(WorkingDirectory, "SpeedTest");
        Utils.ConfigureGlobalLog4netProperties(speedTestWorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(speedTestWorkingDirectory, "log");
        ChallengeTools.Retrain(speedTestWorkingDirectory, "v0", null, 0.9, retrainOnFullDataset: false, useAllAvailableCores: false, computeAndSavePredictions: false, computeValidationRankingScore: false, saveTrainedModel: false);
    }


    private static TransformerNetworkSample GetModelSample()
    {
        var res = new TransformerNetworkSample();
        //res.SetResourceId(-1); //CPU
        return res;
    }
}