using SharpNet.GPU;
using SharpNet.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.HyperParameters;

namespace SharpNet.Networks
{
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    [SuppressMessage("ReSharper", "ConvertToConstant.Local")]
    [SuppressMessage("ReSharper", "FieldCanBeMadeReadOnly.Local")]
    public class Yolov3NetSample : NetworkSample
    {
        #region fields & properties
        private readonly List<Tuple<string, Dictionary<string, string>>> _blocks;
        private static List<Tuple<string, Dictionary<string, string>>> YOLOV3Config => ExtractConfigFileFromContent(Utils.LoadResourceContent(typeof(Yolov3NetSample).Assembly, "SharpNet.ObjectDetection.yolov3.cfg"));
        private readonly IDictionary<int, int> _blockIdToLastLayerIndex = new Dictionary<int, int>();
        private int[] InputShape_CHW = { 3, 608, 608 };
        private double BatchNormMomentum = 0.99;
        private double BatchNormEpsilon = 0.001;
        private double Alpha_LeakyRelu = 0.1;
        
        private float MinScore = 0.5f;
        private float IOU_threshold_for_duplicate = 0.5f;

        private int MaxOutputSize = int.MaxValue;
        private int MaxOutputSizePerClass = int.MaxValue;
        #endregion

        #region constructor

        private Yolov3NetSample([JetBrains.Annotations.NotNull] List<Tuple<string, Dictionary<string, string>>> blocks, ISample[] samples)  : base(samples)
        {
            _blocks = blocks;
        }

        public static Yolov3NetSample ValueOf(List<int> resourceIds, List<Tuple<string, Dictionary<string, string>>> blocks = null)
        {
            var config = new NetworkConfig
                {
                    LossFunction = LossFunctionEnum.CategoricalCrossentropy,
                    CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1,
                    lambdaL2Regularization = 0.0005,
                    WorkingDirectory = Path.Combine(NetworkConfig.DefaultWorkingDirectory, "YOLO"),
                    ResourceIds = resourceIds.ToList()
                }
                .WithSGD(0.9, false)
                .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);

            return new Yolov3NetSample(blocks ?? YOLOV3Config, new ISample[]{config, new DataAugmentationSample()});

        }
        #endregion

        public Network Build()
        {
            LoadNetDescription();
            var network = BuildEmptyNetwork("YOLO V3");

            network.Input(InputShape_CHW[0], InputShape_CHW[1], InputShape_CHW[2], "input_1");

            for (int i = 1; i < _blocks.Count; ++i)
            {
                switch (_blocks[i].Item1)
                {
                    case "convolutional":
                        AddConvolution(network, i);
                        break;
                    case "shortcut":
                        AddShortcut(network, i);
                        break;
                    case "upsample":
                        AddUpSample(network, i);
                        break;
                    case "route":
                        AddRoute(network, i);
                        break;
                    case "yolo":
                        AddYolo(network, i);
                        break;
                }
            }

            var yoloLayers = network.Layers.Where(l => l.GetType() == typeof(YOLOV3Layer)).Select(l => l.LayerIndex).ToArray();
            Debug.Assert(yoloLayers.Length == 3);

            //all predictions of the network
            network.ConcatenateLayer(yoloLayers, "tf_op_layer_concat_6");

            //we remove (set box confidence to 0) low predictions  after non max suppression
            network.NonMaxSuppression(MinScore, IOU_threshold_for_duplicate, MaxOutputSize, MaxOutputSizePerClass, "NonMaxSuppression");

            return network;
        }

        private void LoadNetDescription()
        {
            if (_blocks[0].Item1 != "net")
            {
                throw new ArgumentException("the first item in the config must be 'net'");
            }
            var block = _blocks[0].Item2;
            InputShape_CHW = new[] { int.Parse(block["channels"]), int.Parse(block["height"]), int.Parse(block["width"]) };
            if (block.ContainsKey("momentum"))
            {
                BatchNormMomentum = double.Parse(block["momentum"]);
            }
            _blockIdToLastLayerIndex[0] = 0;
            //TODO: support decay
            //other fields (ignored) : 
            //        batch=64
            //        subdivisions=16
            //        decay=0.0005
            //        angle=0
            //        saturation = 1.5
            //        exposure = 1.5
            //        hue=.1

        }
        private void AddConvolution(Network network, int blockId)
        {
            var block = _blocks[blockId].Item2;
            bool batch_normalize = block.ContainsKey("batch_normalize") && int.Parse(block["batch_normalize"]) >= 1;
            var stride = int.Parse(block["stride"]);
            int lastLayerIndex = _blockIdToLastLayerIndex[blockId - 1];
            if (stride > 1)
            {
                var layerName = GetLayerName(network, "zero_padding2d", typeof(ZeroPadding2DLayer));
                network.ZeroPadding2D(1, 0, 1, 0, lastLayerIndex, layerName);
                lastLayerIndex = network.LastLayerIndex;
            }
            network.Convolution(
                int.Parse(block["filters"]),
                int.Parse(block["size"]),
                stride,
                (stride > 1) ? ConvolutionLayer.PADDING_TYPE.VALID:ConvolutionLayer.PADDING_TYPE.SAME,
                0.0, //lambdaL2Regularization
                !batch_normalize,
                lastLayerIndex,
                "conv_" + (blockId - 1));
            if (batch_normalize)
            {
                network.BatchNorm(BatchNormMomentum, BatchNormEpsilon, "bnorm_" + (blockId - 1));
            }
            if (block.ContainsKey("activation") && !block["activation"].Equals("linear"))
            {
                var activationFunction = ExtractActivation(block["activation"], out var alphaActivation);
                network.Activation(activationFunction, alphaActivation, "leaky_" + (blockId - 1));
            }
            _blockIdToLastLayerIndex[blockId] = network.LastLayerIndex;
        }
        private void AddShortcut(Network network, int blockId)
        {
            var block = _blocks[blockId].Item2;
            var layerName = GetLayerName(network, "tf_op_layer_AddV2", typeof(AddLayer));
            network.AddLayer(_blockIdToLastLayerIndex[blockId - 1], BlockOffsetToLastLayerIndex(int.Parse(block["from"]), blockId), layerName);
            if (block.ContainsKey("activation") && !block["activation"].Equals("linear"))
            {
                var activationFunction = ExtractActivation(block["activation"], out var alphaActivation);
                network.Activation(activationFunction, alphaActivation);
            }
            _blockIdToLastLayerIndex[blockId] = network.LastLayerIndex;
        }
        private void AddUpSample(Network network, int blockId)
        {
            var block = _blocks[blockId].Item2;
            var stride = int.Parse(block["stride"]);
            var layerName = GetLayerName(network, "up_sampling2d", typeof(UpSampling2DLayer));
            network.UpSampling2D(stride, stride, UpSampling2DLayer.InterpolationEnum.Nearest, layerName);
            _blockIdToLastLayerIndex[blockId] = network.LastLayerIndex;
        }
        private void AddYolo(Network network, int blockId)
        {
            var block = _blocks[blockId].Item2;
            var layerName = GetLayerName(network, "yolo", typeof(YOLOV3Layer));
            var selectedAnchorId = ExtractIntArray(block["mask"]);
            var anchors = ExtractIntArray(block["anchors"]);
            var selectedAnchors = new List<int>();
            foreach (var m in selectedAnchorId)
            {
                selectedAnchors.Add(anchors[2*m]);
                selectedAnchors.Add(anchors[2*m+1]);
            }
            network.YOLOV3Layer(selectedAnchors.ToArray(), _blockIdToLastLayerIndex[blockId-1], layerName);
            _blockIdToLastLayerIndex[blockId] = network.LastLayerIndex;
        }
        private void AddRoute(Network network, int blockId)
        {
            var block = _blocks[blockId].Item2;
            var layerIndexes = ExtractIntArray(block["layers"]).Select(blockOffset => BlockOffsetToLastLayerIndex(blockOffset, blockId)).ToArray();
            switch(layerIndexes.Length)
            {
                case 1:
                    _blockIdToLastLayerIndex[blockId] = layerIndexes[0];
                    return;
                case 2:
                    network.ConcatenateLayer(layerIndexes[0], layerIndexes[1], GetLayerName(network, "tf_op_layer_concat", typeof(ConcatenateLayer)));
                    _blockIdToLastLayerIndex[blockId] = network.LastLayerIndex;
                    return;
                default:
                    throw new ArgumentException("invalid route layers " + block["layers"]);
            }
        }
        private static int[] ExtractIntArray(string str)
        {
            return str.Trim().Split(",", StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim()).Select(int.Parse).ToArray();
        }
        private static string GetLayerName(Network network, string layerPrefix, Type layerType)
        {
            int nbExistingAddLayer = network.NbLayerOfType(layerType);
            if (nbExistingAddLayer >= 1)
            {
                return layerPrefix + "_" + nbExistingAddLayer;
            }
            return layerPrefix;
        }
        private int BlockOffsetToLastLayerIndex(int offset, int blockId)
        {
            if (offset >= 0)
            {
                return _blockIdToLastLayerIndex[offset + 1];
            }
            return _blockIdToLastLayerIndex[blockId + offset];
        }
        private cudnnActivationMode_t ExtractActivation(string activationName, out Tensor activationParameter)
        {
            switch (activationName)
            {
                case "leaky":
                    activationParameter = Tensor.SingleFloat((float)Alpha_LeakyRelu);
                    return cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU;
                default:
                    throw new ArgumentException("invalid activation function: " + activationName);
            }

        }
        //public static List<Tuple<string, Dictionary<string, string>>> ExtractConfigFileFromPath(string cfgFilePath)
        //{
        //    return ExtractConfigFileFromContent(File.ReadAllText(cfgFilePath));
        //}
        private static List<Tuple<string, Dictionary<string, string>>> ExtractConfigFileFromContent(string cfgFileContent)
        {
            //"C:\Projects\darknet\cfg\yolov3.cfg"
            var result = new List<Tuple<string, Dictionary<string, string>>>();
            string sectionName = "";
            var sectionContent = new Dictionary<string, string>();
            foreach (var l in cfgFileContent.Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries))
            {
                var trimmed = l.Trim();
                if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#"))
                {
                    continue;
                }

                if (trimmed.StartsWith("["))
                {
                    if (sectionContent.Count != 0)
                    {
                        result.Add(Tuple.Create(sectionName, sectionContent));
                    }
                    sectionContent = new Dictionary<string, string>();
                    sectionName = trimmed.Trim('[', ']').Trim();
                    continue;
                }

                int idx = trimmed.IndexOf('=');
                if (idx >= 0)
                {
                    var key = trimmed.Substring(0, idx).Trim();
                    var value = trimmed.Substring(idx + 1).Trim();
                    sectionContent[key] = value;
                    continue;
                }

                Console.WriteLine("ignoring line: " + Environment.NewLine + l);
            }

            if (sectionContent.Count != 0)
            {
                result.Add(Tuple.Create(sectionName, sectionContent));
            }
            return result;
        }
    }
}