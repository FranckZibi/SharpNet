using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using log4net;
using SharpNet.CPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;

namespace SharpNet.Networks.Transformers;

public static class TextTransformersUtils
{
    public const string NAME = "TextTransformers";


    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(TextTransformersUtils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    // ReSharper disable once MemberCanBePrivate.Global

    //public static string XTrainPath => Path.Combine(DataDirectory, "input_gpt_dev.txt");
    public static string XTrainPath => Path.Combine(DataDirectory, "victor_hugo_v1.txt");


    public static void LoadModel()
    {

        var nn = Network.LoadTrainedNetworkModel(WorkingDirectory, "0604A7130A");
        var outputShape = nn.YPredicted_MiniBatch_Shape(1);
        var max_length = outputShape[1];
        var vocab_size = outputShape[2];
        var datasetSample = new CharLevelTransformersDatasetSample() { max_length = max_length, vocab_size = vocab_size };
        var tokenizer = datasetSample.GetTokenizer();

        var xInputSingleRow = new CpuTensor<float>(new[] { 1, max_length});
        var xInputSingleRowSpan = xInputSingleRow.SpanContent;

        var fulltext = datasetSample.GetFullText();
        var r = new System.Random();
        int randomStartIdx = r.Next(fulltext.Length/2 - max_length-1);

        List<int> tmpSequence = tokenizer.TextsToSequences(new[] { fulltext.Substring(randomStartIdx, 10*max_length) })[0].Take(max_length).ToList();

        int[] newSequence = new int[2000];
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
            var indexNextToken = GetIndexPrediction(proba, r);
            newSequence[nextIndexToGenerate] = indexNextToken;
            ++nextIndexToGenerate;
        }
        var generateText  =tokenizer.SequenceToText(newSequence);
        Console.WriteLine(generateText);
    }


    public static int GetIndexPrediction(float[] proba, Random r)
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
            if (probaWithIndex[i].Item1 > (1.0*probaWithIndex[0].Item1-0.15))
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
    



    public static void LaunchNeuralNetworkHPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //Dataset specific
            //{ "KFold", 3 },
            {"PercentageInTraining", new[]{0.9}},

            {"LossFunction", "SparseCategoricalCrossentropy"},
            {"CompatibilityMode", "TensorFlow"},

            {"max_length", 32},
            {"embedding_dim", 64},

            //{"max_length", 256},
            //{"embedding_dim", 384},

            {"layer_norm_epsilon", 1e-5},
            
            {"layer_norm_before_last_dense", true},
            {"encoder_add_layer_norm_before_mha", true},
            {"encoder_add_layer_norm_after_mha", false},
            {"layer_norm_before_ffd", true},
            {"layer_norm_after_ffd", false},
            
            {"encoder_num_transformer_blocks", 4},
            {"encoder_num_heads", 4},
            
            {"encoder_mha_use_bias_Q_V_K", false},
            {"encoder_mha_use_bias_O", true},
            
            {"encoder_mha_dropout", 0.2f},
            {"encoder_feed_forward_dim", 4*64},
            {"encoder_feed_forward_dropout", 0.2f},
            
            {"encoder_use_causal_mask", true},
            
            //{"vocab_size", 58},   //shakespeare
            {"vocab_size", 81},     // victor hugo 

            
            // Optimizer 
            { "OptimizerType", "AdamW" },
            //{ "AdamW_L2Regularization", 0.01},

            // Learning Rate
            { "InitialLearningRate", new[]{ 0.01 } },
            // Learning Rate Scheduler
            //{ "LearningRateSchedulerType", new[] { "CyclicCosineAnnealing", "OneCycle", "Linear" } },
            { "LearningRateSchedulerType", "CyclicCosineAnnealing"},
            //{ "LearningRateSchedulerType", "Constant"},
            
            { "BatchSize", 64 },

            { "NumEpochs", numEpochs },
            

        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new TransformerNetworkSample(), new CharLevelTransformersDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, false, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

}