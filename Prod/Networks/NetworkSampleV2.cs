//using SharpNet.HyperParameters;
//using SharpNet.Models;

//namespace SharpNet.Networks;

//public class NetworkSampleV2 : NetworkSample
//{

//    // ReSharper disable once MemberCanBePrivate.Global
//    public NetworkSampleV2(ISample[] samples) : base(samples)
//    {
//    }


//    //public override void ApplyDataset(AbstractDatasetSample datasetSample)
//    //{
//    //    var inputShapeOfSingleElement = datasetSample.GetInputShapeOfSingleElement();
//    //    FeatureCounts = inputShapeOfSingleElement[0];
//    //    datasetSampleNumClass = datasetSample.NumClass;
//    //    datasetSampleActivationForLastLayer = datasetSample.ActivationForLastLayer;
//    //}

//    //public static NetworkSampleV2 New(DataSetV2 dataset)
//    //{
//    //    var datasetSample = dataset.DatasetSample;
//    //    var config = new NetworkSampleV2()
//    //    {
//    //        FeatureCounts = dataset.XDataFrame.Shape[1],
//    //        LossFunction = datasetSample.DefaultLossFunction,

//    //        ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST,
//    //        //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, dataset.Name),
//    //        NumEpochs = 10,
//    //        BatchSize = 256,
//    //        InitialLearningRate = 1e-3,
//    //        Metrics = new List<EvaluationMetricEnum> { datasetSample.DefaultLossFunction, datasetSample.GetRankingEvaluationMetric() },
//    //        //ResourceIds = new List<int> {-1}, //on single CPU
//    //        ResourceIds = new List<int> { 0 }, //on single GPU
//    //        AutoSaveIntervalInMinutes = -1,
//    //        DisableReduceLROnPlateau = false,
//    //        CompatibilityMode = NetworkSample.CompatibilityModeEnum.SharpNet,
//    //        RandomizeOrder = true,
//    //    };

//    //    //no Data augmentation
//    //    var da = new DataAugmentationSample();

//    //    return new NetworkSampleV2(new ISample[] { config, da });
//    //}

//    // ReSharper disable once MemberCanBePrivate.Global
//    // ReSharper disable once UnusedMember.Global
//    public NetworkSample_1DCNN HyperParameters => (NetworkSample_1DCNN)Samples[0];
//    // ReSharper disable once MemberCanBePrivate.Global

//    public static NetworkSampleV2 ValueOfNetworkSampleV2(string workingDirectory, string modelName)
//    {
//        return new NetworkSampleV2(new ISample[]
//        {
//            ISample.LoadSample<NetworkSample_1DCNN>(workingDirectory, ISample.SampleName(modelName, 0)),
//            //ISample.LoadSample<DataAugmentationSample>(workingDirectory, ISample.SampleName(modelName, 1)),
//        });
//    }

//    public override void SaveExtraModelInfos(Model model, string workingDirectory, string modelName)
//    {
//        //var n = (Network)model;
//        //var embeddedLayers = n.Layers.Where(t => t is EmbeddingLayer).ToArray();
//        //if (embeddedLayers.Length == 0)
//        //{
//        //    return;
//        //}

//        //var embeddingLayer = ((EmbeddingLayer)embeddedLayers[0]);
//        //var allEmbeddings = embeddingLayer.Split(embeddingLayer.Weights);
//        //for (int i = 0; i < allEmbeddings.Count; ++i)
//        //{
//        //    var df = DataFrame.New(allEmbeddings[i].ToCpuFloat());
//        //    if (df.CellCount != 0)
//        //    {
//        //        df.to_csv(Path.Combine(workingDirectory, model.ModelName + "_embedding" + embeddingLayer.NbForwardPropagation + "_" + i + ".csv"));
//        //    }
//        //}
//    }


//}