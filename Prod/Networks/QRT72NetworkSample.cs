using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Datasets.QRT72;
using SharpNet.HyperParameters;
using SharpNet.Layers;
using SharpNet.Models;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks;

public class QRT72NetworkSample : NetworkSample
{

    private QRT72NetworkSample(ISample[] samples) : base(samples)
    {
    }

    private QRT72NetworkSample(QRT72HyperParameters hyperParameters):
        this(new ISample[] { GetNetworkConfig(hyperParameters), new DataAugmentationSample(), hyperParameters })
    {
    }


    public QRT72HyperParameters QRT72HyperParameters => (QRT72HyperParameters)Samples[2];

    public static QRT72NetworkSample ValueOfQRT72NetworkSample(string workingDirectory, string modelName)
    {
        return new QRT72NetworkSample(new ISample[]
        {
            ISample.LoadSample<NetworkConfig>(workingDirectory, ISample.SampleName(modelName, 0)),
            ISample.LoadSample<DataAugmentationSample>(workingDirectory, ISample.SampleName(modelName, 1)),
            ISample.LoadSample<QRT72HyperParameters>(workingDirectory, ISample.SampleName(modelName, 2))
        });
    }


    public override void SaveExtraModelInfos(IModel model, string workingDirectory, string modelName)
    {
        var n = (Network)model;
        var denseLayers = n.Layers.Where(t => t is DenseLayer).ToArray();
        var A = ((DenseLayer)denseLayers[0]).Weights;
        var beta = ((DenseLayer)denseLayers[1]).Weights;

        var submissionPath = Path.Combine(workingDirectory, modelName + "_submission.csv");

        //We ensure that the matrix is orthogonal (max allowed error : 1e-6)
        var maxError = A.MaxErrorIfOrthogonalMatrix();
        IModel.Log.Info($"Observed Error for orthogonal Matrix: {maxError}");
        if (Math.Abs(maxError) > 1e-6)
        {
            var errorMsg = $"Max Error is too big: {maxError}";
            IModel.Log.Error(errorMsg);
            Console.WriteLine(errorMsg);
            File.WriteAllText(submissionPath, errorMsg);
            return;
        }

        var sb = new StringBuilder();
        sb.Append(",0" + Environment.NewLine);
        int nextIdx = 0;
        foreach (var a in A.ContentAsFloatArray().Concat(beta.ContentAsFloatArray()))
        {
            sb.Append(nextIdx + "," + a.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
            ++nextIdx;
        }
        File.WriteAllText(submissionPath, sb.ToString());

    }


    public static void Run(QRT72HyperParameters hyperParameters)
    {
        var networkSample = new QRT72NetworkSample(hyperParameters);
        var datasetSample = new QRT72DatasetSample(hyperParameters);
        var modelAndDatasetPredictionsSample = ModelAndDatasetPredictionsSample.New(networkSample, datasetSample);
        var modelAndDatasetPredictions = ModelAndDatasetPredictions.New(modelAndDatasetPredictionsSample, QRT72Utils.WorkingDirectory);
        modelAndDatasetPredictions.Fit(true, true, true);
    }


    public override void BuildNetwork(Network network)
    {
        network.Description = Config.ModelName;
        //const cudnnActivationMode_t activationFunction = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;
        network.Input(QRT72Utils.D, -1, -1);
        //network.Dense(QRT72Utils.F, QRT72HyperParameters.lambdaL2Regularization_matrix, false, Optimizer.OptimizationEnum.VanillaSGD);
        //network.Dense(QRT72Utils.F, QRT72HyperParameters.lambdaL2Regularization_matrix, false, Optimizer.OptimizationEnum.VanillaSGDOrtho);
        network.Dense(QRT72Utils.F, QRT72HyperParameters.lambdaL2Regularization_matrix, false);
        network.Layers.Last().DisableBias();
        //network.Dropout(0.2);
        //network.Dense(1, QRT72HyperParameters.lambdaL2Regularization_vector, false, Optimizer.OptimizationEnum.AdamW);
        network.Dense(1, QRT72HyperParameters.lambdaL2Regularization_vector, false);
        network.Layers.Last().DisableBias();
        network.Dropout(0.5);
        //network.Output(1, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
        //network.PropagationManager.LogPropagation = true;
    }

    private static NetworkConfig GetNetworkConfig(QRT72HyperParameters hyperParameters)
    {
        //!D
        hyperParameters.InitialLearningRate = 0.1 * hyperParameters.BatchSize;
        var config = new NetworkConfig
                {
                    LossFunction = LossFunctionEnum.CosineSimilarity504,
                    //lambdaL2Regularization = hyperParameters.lambdaL2Regularization_matrix,
                    lambdaL2Regularization = 0,
                    WorkingDirectory = Path.Combine(NetworkConfig.DefaultWorkingDirectory, QRT72Utils.NAME),
                    NumEpochs = hyperParameters.NumEpochs, //64k iterations
                    BatchSize = hyperParameters.BatchSize,
                    InitialLearningRate = hyperParameters.InitialLearningRate,
                    Metrics = new() { MetricEnum.CosineSimilarity504 },
                    RandomizeOrder = hyperParameters.RandomizeOrder,
                    RandomizeOrderBlockSize = Tensor.CosineSimilarity504_TimeSeries_Length,
                    CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1

        }
                //.WithSGD()
                //.WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
                //.WithAdamW(1e-5)

            //.WithOneCycleLearningRateScheduler(200, 0.1)
            //.WithCifar10ResNetLearningRateScheduler(true, true, false);
            //.WithConstantLearningRateScheduler(hyperParameters.InitialLearningRate)
            .WithLinearLearningRateScheduler(100)
            ;
        config.ModelName = hyperParameters.ComputeHash();
        //config.DisplayTensorContentStats = config.SaveNetworkStatsAfterEachEpoch = true;

        config.ResourceIds = new() { -1 }; //use CPU

        return config;
    }

    //private const string FILE_SUFFIX = "";
    //private const string FILE_SUFFIX = "_small";


}