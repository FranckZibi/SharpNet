using System.Collections.Generic;
using System.IO;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.Networks;

public class InMemoryDataSetV2NetworkSample : NetworkSample
{

    // ReSharper disable once MemberCanBePrivate.Global
    public InMemoryDataSetV2NetworkSample(ISample[] samples) : base(samples)
    {
    }

    public static InMemoryDataSetV2NetworkSample New(InMemoryDataSetV2 dataset)
    {
        var datasetSample = dataset.DatasetSample;
        var config = new InMemoryDataSetV2NetworkSampleHyperParameters() 
        {
            FeatureCounts = dataset.XDataFrame.Shape[1],
            LossFunction = datasetSample.DefaultLossFunction,
            datasetSampleNumClass = datasetSample.NumClass,
            datasetSampleActivationForLastLayer = datasetSample.ActivationForLastLayer,

            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST,
            WorkingDirectory = Path.Combine(NetworkConfig.DefaultWorkingDirectory, dataset.Name),
            NumEpochs = 10,
            BatchSize = 256,
            InitialLearningRate = 1e-3,
            Metrics = new List<EvaluationMetricEnum> { datasetSample.DefaultLossFunction, datasetSample.GetRankingEvaluationMetric() },
            //ResourceIds = new List<int> {-1}, //on single CPU
            ResourceIds = new List<int> { 0 }, //on single GPU
            AutoSaveIntervalInMinutes = -1,
            DisableReduceLROnPlateau = false,
            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.SharpNet,
            RandomizeOrder = true,
        };

        (config.VocabularySizes, config.EmbeddingDims, config.IndexesInLastDimensionToUse) = dataset.EmbeddingDescription(config.DefaultEmbeddingDim);

        //no Data augmentation
        var da = new DataAugmentationSample();
      
        return new InMemoryDataSetV2NetworkSample(new ISample[] { config, da});
    }

    // ReSharper disable once MemberCanBePrivate.Global
    // ReSharper disable once UnusedMember.Global
    public InMemoryDataSetV2NetworkSampleHyperParameters HyperParameters => (InMemoryDataSetV2NetworkSampleHyperParameters)Samples[0];
    // ReSharper disable once MemberCanBePrivate.Global

    public static InMemoryDataSetV2NetworkSample ValueOfInMemoryDataSetV2NetworkSample(string workingDirectory, string modelName)
    {
        return new InMemoryDataSetV2NetworkSample(new ISample[]
        {
            ISample.LoadSample<InMemoryDataSetV2NetworkSampleHyperParameters>(workingDirectory, ISample.SampleName(modelName, 0)),
            ISample.LoadSample<DataAugmentationSample>(workingDirectory, ISample.SampleName(modelName, 1)),
        });
    }

    public override void CreateLayers(Network network)
    {
        HyperParameters.BuildNetwork(network);
    }


    public override void SaveExtraModelInfos(Model model, string workingDirectory, string modelName)
    {
        //var n = (Network)model;
        //var embeddedLayers = n.Layers.Where(t => t is EmbeddingLayer).ToArray();
        //if (embeddedLayers.Length == 0)
        //{
        //    return;
        //}

        //var embeddingLayer = ((EmbeddingLayer)embeddedLayers[0]);
        //var allEmbeddings = embeddingLayer.Split(embeddingLayer.Weights);
        //for (int i = 0; i < allEmbeddings.Count; ++i)
        //{
        //    var df = DataFrame.New(allEmbeddings[i].ToCpuFloat());
        //    if (df.CellCount != 0)
        //    {
        //        df.to_csv(Path.Combine(workingDirectory, model.ModelName + "_embedding" + embeddingLayer.NbForwardPropagation + "_" + i + ".csv"));
        //    }
        //}
    }


}