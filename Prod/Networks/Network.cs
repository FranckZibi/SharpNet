using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Text;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Optimizers;

namespace SharpNet.Networks
{
    public class Network : IDisposable
    {
        #region private fields
        private readonly Stopwatch _spInternalFit = new Stopwatch();
        private readonly Stopwatch _swComputeLoss;
        private readonly Stopwatch _swComputeAccuracy;
        private readonly List<EpochData> _epochData;
        private readonly IDictionary<string,Stopwatch> _updateWeightsTime = new Dictionary<string, Stopwatch>();
        private Tensor _yPredictedBufferForEntireBatch;
        private Tensor _yExpectedBufferForEntireBatch;
        private CpuTensor<float> _xMiniBatchCpu;
        private CpuTensor<float> _yExpectedMiniBatchCpu;
        private Tensor _bufferComputeAccuracy;
        private Tensor _bufferComputeLoss;
        private readonly DateTime _timeStampCreation = DateTime.Now;
        // bytes/batchSize needed for forward & backward propagation
        private ulong? _bytesByBatchSize;
        private readonly List<int> _gpuDeviceIds;
        #endregion

        #region public fields
        public NetworkConfig Config { get; }
        public List<Layer> Layers { get; } = new List<Layer>();
        public string Description { private get; set; } = "";
        public PropagationManager PropagationManager { get; }
        public IDictionary<string, Stopwatch> ForwardPropagationTrainingTime { get; } = new Dictionary<string, Stopwatch>();
        public IDictionary<string, Stopwatch> ForwardPropagationInferenceTime { get; } = new Dictionary<string, Stopwatch>();
        public IDictionary<string, Stopwatch> BackwardPropagationTime { get; } = new Dictionary<string, Stopwatch>();
        public TensorMemoryPool MemoryPool { get; }
        public GPUWrapper GpuWrapper { get; }
        public int GpuDeviceId { get; }
        #endregion


        /// <param name="config"></param>
        /// <param name="resourceIds">
        /// list of resources available for the network
        /// if gpuDeviceIds.Count == 1
        ///     if masterNetworkIfAny == null:
        ///         all computation will be done in a single network (no parallel computing between different networks
        ///     else:
        ///         we are using several networks in // for doing the computation
        ///         and we are in a slave network doing part of this parallel computation for the master network 'masterNetworkIfAny'
        /// else: (gpuDeviceIds.Count >= 2)
        ///     we are using several networks in // for doing the computation
        ///     and we are in the master network doing part of this parallel computation
        ///     this master network will use resourceId gpuDeviceIds[1]
        ///     slaves network will use resourceId gpuDeviceIds[1:]
        ///
        /// for each resourceId in this list:
        ///     if resourceId strictly less then 0:
        ///         use CPU resource (no GPU usage)
        ///     else:
        ///         run the network on the GPU with device Id = resourceId
        /// </param>
        /// <param name="epochData"></param>
        /// <param name="masterNetworkIfAny">
        /// if the current network is a slave network doing computation for its master network:
        ///     the reference of the master network
        /// else:
        ///     null
        /// </param>
        public Network(NetworkConfig config, List<int> resourceIds, List<EpochData> epochData = null, Network masterNetworkIfAny = null)
        {
            //a slave network will have access to only one resource (Cpu or GPU device id)
            Debug.Assert(masterNetworkIfAny == null || resourceIds.Count == 1);
            Config = config;
            _epochData = epochData ?? new List<EpochData>();
            _gpuDeviceIds = resourceIds;
            _masterNetworkIfAny = masterNetworkIfAny;
            GpuDeviceId = _gpuDeviceIds[0];
            GpuWrapper = UseGPU ? GPUWrapper.FromDeviceId(GpuDeviceId) : null;
            _swComputeLoss = new Stopwatch();
            _swComputeAccuracy = new Stopwatch();
            CreateLogDirectoryIfNeeded();
            MemoryPool = new TensorMemoryPool(GpuWrapper, false);
            PropagationManager = new PropagationManager(Layers, MemoryPool, ForwardPropagationTrainingTime, ForwardPropagationInferenceTime, BackwardPropagationTime, _updateWeightsTime);
            _xMiniBatchCpu = new CpuTensor<float>(new[] { 1 }, null, nameof(_xMiniBatchCpu));
            _yExpectedMiniBatchCpu = new CpuTensor<float>(new[] { 1 }, null, nameof(_yExpectedMiniBatchCpu));
        }

        #region Master Network data
        private bool IsMaster => _masterNetworkIfAny == null;
        /// <summary>
        /// list of all slave networks
        /// empty list of:
        ///     the 'this' network is a slave network
        ///     or
        ///     the 'this' network is a master network  but we are not using multi GPU computation
        ///     (all computation is performed on the master network)
        /// </summary>
        private readonly List<Network> _slaveNetworks = new List<Network>();
        #endregion
        #region Slave Network data
        /// <summary>
        /// if the current network is a slave network doing computation for its master network:
        ///     the reference of the master network
        /// if current network is a master network:
        ///     null
        /// </summary>
        private readonly Network _masterNetworkIfAny;
        #endregion


        #region Transfer Learning
        /// <summary>
        /// set the number of output categories of the current network by updating the head layers (Dense+Activation layers)
        /// if the number of output categories is already 'newCategoryCount'
        ///     does nothing at all
        /// else
        ///     update the last Dense Layers (resetting all its weights) to match the required number of categories
        /// </summary>
        /// <param name="newCategoryCount">the target number of categories</param>
        public void SetCategoryCount(int newCategoryCount)
        {
            if (Layers.Count>=2 && Layers[Layers.Count - 1] is ActivationLayer && Layers[Layers.Count-2] is DenseLayer)
            {
                var denseLayer = (DenseLayer)Layers[Layers.Count - 2];
                if (denseLayer.CategoryCount == newCategoryCount)
                {
                    Info("no need to set the CategoryCount to "+newCategoryCount);
                    return; //already at target category count
                }

                //we remove the ActivationLayer (last layer)
                var activationLayer = (ActivationLayer)Layers.Last();
                var activationFunctionType = activationLayer.ActivationFunction;
                var activationLayerName = activationLayer.LayerName;
                RemoveAndDisposeLastLayer();

                //we remove the Dense layer
                var lambdaL2Regularization = denseLayer._lambdaL2Regularization;
                var denseLayerName = denseLayer.LayerName;
                RemoveAndDisposeLastLayer();

                //We add a new DenseLayer (with weight reseted)
                Info("Resetting weights of layer "+ denseLayerName + " to have " + newCategoryCount+" categories");
                Dense(newCategoryCount, lambdaL2Regularization, denseLayerName);

                //we put back the ActivationLayer
                Activation(activationFunctionType, activationLayerName);

                return;
            }
            throw new NotImplementedException("can only update a network where the 2 last layers are DenseLayer & ActivationLayer");
        }
        private void RemoveAndDisposeLastLayer()
        {
            var lastLayer = Layers.Last();
            foreach (var previousLayers in lastLayer.PreviousLayers)
            {
                previousLayers.NextLayerIndexes.Remove(lastLayer.LayerIndex);
            }
            lastLayer.Dispose();
            Layers.RemoveAt(Layers.Count - 1);
            OnLayerAddOrRemove();
        }
        #endregion

        /// <summary>
        /// Clone the current network
        /// </summary>
        /// <param name="slaveDeviceId"></param>
        /// <returns></returns>
        public Network CreateSlaveNetwork(int slaveDeviceId)
        {
            var clonedNetwork = new Network(Config, new List<int> { slaveDeviceId }, new List<EpochData>(_epochData));
            clonedNetwork.Description = Description;
            foreach (var l in Layers)
            {
                clonedNetwork.Layers.Add(l.Clone(clonedNetwork));
            }
            return clonedNetwork;
        }
        public string DeviceName() { return GpuWrapper?.DeviceName(); }
        public void Dispose()
        {
            LogDebug("Before clearing memory: " + MemoryInfo());
            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect();
            Layers.ForEach(l => l?.Dispose());
            Layers.Clear();
            PropagationManager.Dispose();
            _epochData.Clear();

            MemoryPool.FreeMemory(ref _bufferComputeAccuracy);
            MemoryPool.FreeMemory(ref _bufferComputeLoss);
            MemoryPool.FreeMemory(ref _yPredictedBufferForEntireBatch);
            MemoryPool.FreeMemory(ref _yExpectedBufferForEntireBatch);

            _xMiniBatchCpu?.Dispose();
            _xMiniBatchCpu = null;

            _yExpectedMiniBatchCpu?.Dispose();
            _yExpectedMiniBatchCpu = null;

            GpuWrapper?.Reset();

            MemoryPool.Dispose();

            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect();
            LogDebug("After clearing memory: " + MemoryInfo());
        }
        /// <summary>
        /// Compares the 'this' network with the 'other' network and write a test report in the 'errors' output field
        /// </summary>
        /// <param name="other">the network to compare with the 'this' network</param>
        /// <param name="errors">field where the report results will be stored</param>
        /// <returns>true if the 2 networks are the same, else if a difference has been found</returns>
        public bool Equals(Network other, out string errors)
        {
            var id = Description;
            errors = "";
            const double epsilon = 1e-5;
            var equals = true;
            equals &= Utils.Equals(Description, other.Description, id + ":Description", ref errors);
            equals &= Config.Equals(other.Config, epsilon, id, ref errors);
            equals &= Utils.Equals(other.GpuDeviceId, GpuDeviceId, id, ref errors);
            equals &= Utils.Equals(Layers.Count, other.Layers.Count, id + ":Layers.Count", ref errors);
            for (int i = 0; i < Math.Min(Layers.Count, other.Layers.Count); ++i)
            {
                equals &= Layers[i].Equals(other.Layers[i], epsilon, id + ":Layers["+i+"]", ref errors);
            }
            return equals;
        }

        #region network construction: adding layers
        public Network Input(int channelCount, int h, int w, string layerName = "")
        {
            Layers.Add(new InputLayer(channelCount, h, w, this, layerName));
            return this;
        }
        public Network SimpleRnnLayer(int xLength, int aLength, int yLength, bool returnSequences, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            var simpleRnnLayer = new SimpleRnnLayer(xLength, aLength, yLength, returnSequences, this, layerName);
            Layers.Add(simpleRnnLayer);
            return this;
        }
        public Network Dense(int categoryCount, double lambdaL2Regularization, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            var fullyConnectedLayer = new DenseLayer(categoryCount, lambdaL2Regularization, this, layerName);
            Layers.Add(fullyConnectedLayer);
            return this;
        }
        public Network Convolution_BatchNorm(int filtersCount, int f, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization)
        {
            return Convolution(filtersCount, f, stride, paddingType, lambdaL2Regularization, false)
                .BatchNorm(0.99, 1e-5);
        }
        public Network Convolution_BatchNorm_Activation(int filtersCount, int f, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, cudnnActivationMode_t activationFunction)
        {
            return Convolution_BatchNorm(filtersCount, f, stride, paddingType, lambdaL2Regularization)
                .Activation(activationFunction);
        }
        public Network BatchNorm_Activation(cudnnActivationMode_t activationFunction)
        {
            return BatchNorm(0.99, 1e-5).Activation(activationFunction);
        }
        public Network BatchNorm_Activation_Convolution(cudnnActivationMode_t activationFunction, int filtersCount, int f, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias)
        {
            return 
                BatchNorm(0.99, 1e-5)
                .Activation(activationFunction)
                .Convolution(filtersCount, f, stride, paddingType, lambdaL2Regularization, useBias);
        }
        public Network AddLayer(int previousIdentityLayerIndex, int previousResidualLayerIndex, string layerName = "")
        {
            Layers.Add(new AddLayer(previousIdentityLayerIndex, previousResidualLayerIndex, this, layerName));
            Debug.Assert(Layers[previousIdentityLayerIndex].SameOutputShape(Layers[previousResidualLayerIndex]));
            return this;
        }
        public Network ConcatenateLayer(int previousLayerIndex1, int previousLayerIndex2, string layerName = "")
        {
            Layers.Add(new ConcatenateLayer(previousLayerIndex1, previousLayerIndex2, this, layerName));
            return this;
        }
        public Network MultiplyLayer(int previousLayerIndex1, int previousLayerIndex2, string layerName = "")
        {
            Layers.Add(new MultiplyLayer(previousLayerIndex1, previousLayerIndex2, this, layerName));
            return this;
        }
        //add a shortcut from layer 'AddSumLayer' to current layer, adding a Conv Layer if necessary (for matching size)
        public Network Shortcut_IdentityConnection(int startOfBlockLayerIndex, int filtersCount, int stride, double lambdaL2Regularization)
        {
            int previousResidualLayerIndex = LastLayerIndex;

            var sameInputAndOutputShapeInBlock = Layers.Last().SameOutputShape(Layers[startOfBlockLayerIndex]);
            if (sameInputAndOutputShapeInBlock)
            {
                Layers.Add(new AddLayer(startOfBlockLayerIndex, previousResidualLayerIndex, this));
            }
            else
            {
                //we need to add a convolution layer to make correct output format
                Convolution(filtersCount, 1, stride, 0, lambdaL2Regularization, true, startOfBlockLayerIndex);
                int convLayerIdInIdentityBlock = LastLayerIndex;
                Layers.Add(new AddLayer(convLayerIdInIdentityBlock, previousResidualLayerIndex, this));
                Debug.Assert(Layers[convLayerIdInIdentityBlock].SameOutputShape(Layers[previousResidualLayerIndex]));
            }
            return this;
        }
        public Network Convolution(int filtersCount, int f, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, string layerName = "")
        {
            return Convolution(filtersCount, f, stride, paddingType, lambdaL2Regularization, useBias, Layers.Count - 1, layerName);
        }
        public Network Convolution(int filtersCount, int f, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, int previousLayerIndex, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ConvolutionLayer(false, filtersCount, -1, f, stride, paddingType, lambdaL2Regularization, useBias, previousLayerIndex, this, layerName));
            return this;
        }
        public Network DepthwiseConvolution(int f, int stride, ConvolutionLayer.PADDING_TYPE paddingType, int depthMultiplier, double lambdaL2Regularization, bool useBias, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ConvolutionLayer(true, -1, depthMultiplier, f, stride, paddingType, lambdaL2Regularization, useBias, Layers.Count - 1, this, layerName));
            return this;
        }
        public Network Dropout(double dropProbability, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new DropoutLayer(dropProbability, this, layerName));
            return this;
        }
        public Network Activation(cudnnActivationMode_t activationFunction, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ActivationLayer(activationFunction, this, layerName));
            return this;
        }

        public Network MaxPooling(int poolingHeight, int poolingWidth, int poolingStride, string layerName = "")
        {
            return MaxPooling(poolingHeight, poolingWidth, poolingStride, Layers.Count-1, layerName);
        }
        private Network MaxPooling(int poolingHeight, int poolingWidth, int poolingStride, int previousLayerIndex, string layerName)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new PoolingLayer(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingHeight, poolingWidth, poolingStride, previousLayerIndex, this, layerName));
            return this;
        }
        public Network AvgPooling(int poolingHeight, int poolingWidth, int poolingStride, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            int previousLayerIndex = Layers.Count-1;
            Layers.Add(new PoolingLayer(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, poolingHeight, poolingWidth, poolingStride, previousLayerIndex, this, layerName));
            return this;
        }
        public Network GlobalAvgPooling(string layerName = "")
        {
            var lastLayerShape = Layers.Last().OutputShape(1);
            var lastLayerHeight = lastLayerShape[2];
            var lastLayerWidth = lastLayerShape[3];
            var poolingStride = Math.Max(lastLayerHeight, lastLayerWidth);
            return AvgPooling(lastLayerHeight, lastLayerWidth, poolingStride, layerName);
        }

        public Network GlobalMaxPooling(string layerName = "")
        {
            return GlobalMaxPooling(Layers.Count - 1, layerName);
        }

        public Network GlobalMaxPooling(int previousLayerIndex, string layerName = "")
        {
            var previousLayerShape = Layers[previousLayerIndex].OutputShape(1);
            var previousLayerHeight = previousLayerShape[2];
            var previousLayerWidth = previousLayerShape[3];
            var poolingStride = Math.Max(previousLayerHeight, previousLayerWidth);
            return MaxPooling(previousLayerHeight, previousLayerWidth, poolingStride, previousLayerIndex, layerName);
        }
        public Network GlobalAvgPooling_And_GlobalMaxPooling()
        {
            GlobalAvgPooling();
            int globalAvgPoolingLayerIndex = LastLayerIndex;
            GlobalMaxPooling(LastLayerIndex);
            int globalMaxPoolingLayerIndex = LastLayerIndex;
            return ConcatenateLayer(globalAvgPoolingLayerIndex, globalMaxPoolingLayerIndex);
        }

        public Network BatchNorm(double momentum, double epsilon, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new BatchNormalizationLayer(momentum, epsilon, this, layerName));
            return this;
        }
        public Network Dense_Activation(int n_x, double lambdaL2Regularization, cudnnActivationMode_t activationFunction)
        {
            return Dense(n_x, lambdaL2Regularization)
                .Activation(activationFunction);
        }
        public Network Dense_DropOut_Activation(int n_x, double lambdaL2Regularization, double dropOut, cudnnActivationMode_t activationFunction)
        {
            return Dense(n_x, lambdaL2Regularization)
                .Dropout(dropOut)
                .Activation(activationFunction);
        }
        public Network Output(int n_x, double lambdaL2Regularization, cudnnActivationMode_t activationFunctionType)
        {
            return Dense(n_x, lambdaL2Regularization)
                .Activation(activationFunctionType);
        }
        public Network Flatten()
        {
            Debug.Assert(Layers.Count >= 1);
            var flattenLayer = new FlattenLayer(this);
            Layers.Add(flattenLayer);
            return this;
        }
        #endregion

        public override string ToString()
        {
            var result = Summary() + Environment.NewLine;
            var xShape = (Layers.Count > 0)?Layers[0].OutputShape(1): new[] { 1, 1, 1, 1 };
            var bytesByBatchSizeForwardAndBackward = BytesByBatchSizeForwardAndBackward(xShape, true);
            result += Utils.MemoryBytesToString(bytesByBatchSizeForwardAndBackward) + "/batchSize + "+
                      Utils.MemoryBytesToString(BytesIndependentOfBatchSize);
            return result;
        }
        public Optimizer GetOptimizer(int[] weightShape, int[] biasShape)
        {
            switch (Config.OptimizerType)
            {
                case Optimizer.OptimizationEnum.Adam: return new Adam(this, Config.Adam_beta1, Config.Adam_beta2, Config.Adam_epsilon, weightShape, biasShape);
                case Optimizer.OptimizationEnum.SGD: return new Sgd(this, Config.SGD_momentum, Config.SGD_usenesterov, weightShape, biasShape);
                default: return VanillaSgd.Instance;
            }
        }
        public List<Tensor> TensorsIndependentOfBatchSize
        {
            get { return Layers.SelectMany(x => x.TensorsIndependentOfBatchSize).Where(x => x != null).ToList(); }
        }
        public int TotalParams => Layers.Select(x => x.TotalParams).Sum();

        #region serialization
        // ReSharper disable once UnusedMember.Global
        public static Network ValueOf(string path, int?overrideGpuDeviceId = null)
        {
            var allLines = File.ReadAllLines(path);
            var dicoFirstLine = Serializer.Deserialize(allLines[0], null);
            var config = NetworkConfig.ValueOf(dicoFirstLine);
            var gpuDeviceId = overrideGpuDeviceId ?? (int)dicoFirstLine[nameof(GpuDeviceId)];
            var epochsData = (EpochData[])dicoFirstLine[nameof(_epochData)];
            var network = new Network(config, new List<int>{gpuDeviceId}, epochsData.ToList());
            network.Description = dicoFirstLine.TryGet<string>(nameof(Description))??"";
            for (int i = 1; i < allLines.Length; ++i)
            {
                network.Layers.Add(Layer.ValueOf(Serializer.Deserialize(allLines[i], network.GpuWrapper), network));
            }
            return network;
        }
        public void Save(string fileName = "")
        {
            if (string.IsNullOrEmpty(fileName))
            {
                fileName = Path.Combine(Config.LogDirectory, UniqueId + ".txt");
            }
            var firstLine = new Serializer()
                .Add(nameof(Description), Description)
                .Add(Config.Serialize())
                .Add(nameof(GpuDeviceId), GpuDeviceId)
                .Add(nameof(_epochData), _epochData.ToArray())
                .ToString();
            File.AppendAllLines(fileName, new[] { firstLine });
            foreach (var l in Layers)
            {
                File.AppendAllLines(fileName, new[] { l.Serialize() });
            }
        }
        public void LogContent()
        {
            for (int layerIndex = 0; layerIndex < Layers.Count; ++layerIndex)
            {
                //if (layerIndex > 10 && layerIndex < Layers.Count - 10) continue;
                var layer = Layers[layerIndex];
                layer.LogContent();
            }
        }
        #endregion

        [SuppressMessage("ReSharper", "UnusedParameter.Local")]
        private void CheckInput(IDataSet trainingDateSetCpu, IDataSet testDateSetCpu, ILearningRateComputer learningRateComputer, int numEpochs, int miniBatchSize)
        {
            if (trainingDateSetCpu.TypeSize != Config.TypeSize)
            {
                throw new Exception("Invalid type : expecting: "+Config.TypeSize+" but was "+ trainingDateSetCpu.TypeSize);
            }
            foreach (var l in Layers)
            {
                l.CheckConsistency();
            }
            
        }

        public double FindBestLearningRate(IDataSet trainingDataSet, double minLearningRate, double maxLearningRate, int miniBatchSize = -1)
        {
            Info("Looking for best learning rate...");
            ResetWeights(); //restore weights to there original values
            var maxMiniBatchSize = MaxMiniBatchSize(trainingDataSet.XMiniBatch_Shape(1), false);
            if (miniBatchSize < 1)
            {
                miniBatchSize = maxMiniBatchSize;
            }
            else
            {
                if (miniBatchSize > maxMiniBatchSize)
                {
                    Info("Reducing BatchSize from "+miniBatchSize+" to "+maxMiniBatchSize);
                    miniBatchSize = maxMiniBatchSize;
                }
            }
            var learningRateFinder = new LearningRateFinder(miniBatchSize, trainingDataSet.Count, minLearningRate, maxLearningRate);
                

            bool CallBackAfterEachMiniBatch(Tensor yExpectedMiniBatch, Tensor yPredictedMiniBatch, int blockIdInEpoch, int nbBatchBlockInEpoch, int epoch)
            {
                MemoryPool.GetNotInitializedFloatTensor(ref _bufferComputeLoss, new[] { yExpectedMiniBatch.Shape[0] }, nameof(_bufferComputeLoss));
                var blockLoss = yExpectedMiniBatch.ComputeLoss(yPredictedMiniBatch, Config.LossFunction, _bufferComputeLoss);
                return learningRateFinder.AddLossForLastBlockId(blockLoss);
            }
            MiniBatchGradientDescent(trainingDataSet, miniBatchSize, learningRateFinder, CallBackAfterEachMiniBatch);
            var fileName = Path.Combine(Config.LogDirectory, UniqueId + "_LearningRateFinder.csv");
            File.WriteAllText(fileName, learningRateFinder.AsCsv());
            Info("Stats stored in: " + fileName);
            var bestLearningRate = learningRateFinder.BestLearningRate();
            Info("Best learning rate: "+ bestLearningRate+ " (with batch size="+miniBatchSize+")");
            ResetWeights(); //restore weights to there original values
            return bestLearningRate;
        }

        private string MiniBatchLossFile => Path.Combine(Config.LogDirectory, UniqueId + "_MiniBatchLoss.csv");

        bool CallBackComputeLossAfterEachMiniBatch(Tensor yExpectedMiniBatch, Tensor yPredictedMiniBatch, int blockIdInEpoch, int nbBatchBlockInEachEpoch, int epoch)
        {
            var fileName = MiniBatchLossFile;
            if (!File.Exists(fileName))
            {
                File.WriteAllText(fileName, "Sep=;"+Environment.NewLine+"Epoch;Iteration;Loss"+Environment.NewLine);
            }
            _swComputeLoss?.Start();
            MemoryPool.GetNotInitializedFloatTensor(ref _bufferComputeLoss, new[] { yExpectedMiniBatch.Shape[0] }, nameof(_bufferComputeLoss));
            var blockLoss = yExpectedMiniBatch.ComputeLoss(yPredictedMiniBatch, Config.LossFunction, _bufferComputeLoss);
            _swComputeLoss?.Stop();
            int iteration = (epoch - 1) * nbBatchBlockInEachEpoch + blockIdInEpoch;
            File.AppendAllText(fileName, epoch+";"+ iteration + ";"+blockLoss.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
            return false;
        }

        public void FreezeSelectedLayers()
        {
            if (string.IsNullOrEmpty(Config.FirstLayerNameToFreeze) && string.IsNullOrEmpty(Config.LastLayerNameToFreeze))
            {
                //no layers to freeze : we'll train the entire network
                return; 
            }
            var firstLayerIndexToFreeze = LayerNameToLayerIndex(Config.FirstLayerNameToFreeze);
            if (firstLayerIndexToFreeze == -1)
            {
                firstLayerIndexToFreeze = 0;
            }
            var lastLayerIndexToFreeze = LayerNameToLayerIndex(Config.LastLayerNameToFreeze);
            if (lastLayerIndexToFreeze == -1)
            {
                lastLayerIndexToFreeze = LastLayerIndex;
            }

            Info("Freezing " + (lastLayerIndexToFreeze - firstLayerIndexToFreeze + 1) + " layers (between " + Layers[firstLayerIndexToFreeze].LayerName + " and " + Layers[lastLayerIndexToFreeze].LayerName + ")");
            for (int layerIndex = 0; layerIndex < Layers.Count; ++layerIndex)
            {
                var layer = Layers[layerIndex];
                if (layerIndex >= firstLayerIndexToFreeze && layerIndex <= lastLayerIndexToFreeze)
                {
                    //we need to freeze the weights/bias associated with the layer
                    layer.Trainable = false;
                }
                else
                {
                    //the layer is trainable
                    layer.Trainable = true;
                    //we reset the layer weights to their default values
                    layer.ResetWeights();
                }
            }
        }

        /// <summary>
        /// Here is a summary of what is used at each step ('forward pass' / 'backward pass' / 'weights update' of the training
        /// 'X'  : input tensor of each layer
        /// 'dX' : gradient of the input tensor
        /// 'Y'  : output tensor of each layer
        /// 'dY' : gradient of the output tensor
        /// 'W'  : weights and bias
        /// 'dW' : gradient of weights and bias
        /// ===================================================================
        /// Step           =  'X'  = 'dX'  = 'Y'   = 'dY'  = 'W'      = 'dW'  =
        /// ===================================================================
        /// Forward Pass   = [in]  =       = [out] =       = [in]     =       =
        /// Backward Pass  = [in]  = [out] = [in]  = [in]  = [in]     = [out] =
        /// Weights Update =       =       =       =       = [in,out] = [in]  =
        /// ===================================================================
        /// </summary>
        /// <param name="trainingDataSetCpu"></param>
        /// <param name="learningRateComputer"></param>
        /// <param name="numEpochs"></param>
        /// <param name="preferredMiniBatchSize"></param>
        /// <param name="testDataSetCpuIfAny"></param>
        public void Fit(IDataSet trainingDataSetCpu, ILearningRateComputer learningRateComputer, int numEpochs, int preferredMiniBatchSize, IDataSet testDataSetCpuIfAny)
        {
            try
            {
                Debug.Assert(Config.TypeSize == trainingDataSetCpu.TypeSize);
                Debug.Assert(learningRateComputer != null);
                _spInternalFit.Start();
                StartTimer("Fit_Prepare", ForwardPropagationTrainingTime);

                FreezeSelectedLayers();
                CheckInput(trainingDataSetCpu, testDataSetCpuIfAny, learningRateComputer, numEpochs, preferredMiniBatchSize);
                
                Info(ToString());
                var maxMiniBatchSize = MaxMiniBatchSize(trainingDataSetCpu.XMiniBatch_Shape(1), true);
                var miniBatchSize = preferredMiniBatchSize;
                if (miniBatchSize < 1)
                {
                    miniBatchSize = maxMiniBatchSize;
                    Info("Using (auto) MiniBatchSize of " + miniBatchSize);
                }
                else if (miniBatchSize > maxMiniBatchSize)
                {
                    Info("Reducing MiniBatchSize from "+ miniBatchSize+" to "+ maxMiniBatchSize+" because of memory limit.");
                    miniBatchSize = maxMiniBatchSize;
                }


                var nbBlocksInEpoch = NbBlocksInEpoch(miniBatchSize, trainingDataSetCpu.Count);
                if (UseGPU)
                {
                    LogDebug(GpuWrapper.ToString());
                }
                LogDebug("Training Set: " + trainingDataSetCpu);
                if (testDataSetCpuIfAny != null)
                {
                    LogDebug("Test Set: " + testDataSetCpuIfAny);
                }
                Info("#Epochs=" + numEpochs + " BatchSize=" + miniBatchSize+" Name="+Description);
                if (Config.DisplayTensorContentStats)
                {
                    LogDebug("Initial Tensor Content stats" + Environment.NewLine + ContentStats() + Environment.NewLine);
                }

                Func<Tensor, Tensor, int, int, int, bool> callBackAtEachIteration = null;
                if (Config.SaveLossAfterEachMiniBatch)
                {
                    Info("Saving mini batch loss in " + MiniBatchLossFile);
                    callBackAtEachIteration = CallBackComputeLossAfterEachMiniBatch;
                }

                //Info(GpuWrapper.ToString());

                StopTimer("Fit_Prepare", ForwardPropagationTrainingTime);


                var lastAutoSaveTime = DateTime.Now; //last time we saved the network
                Tuple<double, double> validationLossAndAccuracy = null;
                for (;;)
                {
                    int epoch = _epochData.Count + 1;
                    if (epoch > numEpochs)
                    {
                        break;
                    }

                    var swEpoch = Stopwatch.StartNew();

                    var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputer.MultiplicativeFactorFromReduceLrOnPlateau(_epochData);
                    if (learningRateComputer.ShouldReduceLrOnPlateau(_epochData))
                    {
                        Info("Reducing learningRate because of plateau at epoch " + epoch + " (new multiplicative coeff:"+ lrMultiplicativeFactorFromReduceLrOnPlateau+")");
                    }

                    #region Mini Batch gradient descent
                    var learningRateAtEpochStart = learningRateComputer.LearningRate(epoch, 0, nbBlocksInEpoch, lrMultiplicativeFactorFromReduceLrOnPlateau);
                    var yPredicted = MiniBatchGradientDescent(trainingDataSetCpu, miniBatchSize, learningRateComputer, callBackAtEachIteration);
                    #endregion

                    //We display stats about the just finished epoch
                    if (Config.DisplayTensorContentStats)
                    {
                        LogDebug("End of Epoch:" + epoch + " Tensor Content stats" + Environment.NewLine+ContentStats()+Environment.NewLine);
                    }

                    StartTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);
                    var trainLossAndAccuracyForEpoch = ComputeLossAndAccuracyForEntireBatch(_yExpectedBufferForEntireBatch, yPredicted);
                    var lossAndAccuracyMsg = LossAndAccuracyToString(trainLossAndAccuracyForEpoch, "");
                    if (testDataSetCpuIfAny != null)
                    {
                        //We compute the validation (= test) loss&accuracy
                        validationLossAndAccuracy = ComputeLossAndAccuracyForTestDataSet(miniBatchSize, testDataSetCpuIfAny);
                        lossAndAccuracyMsg += " - "+LossAndAccuracyToString(validationLossAndAccuracy, "val_");
                    }
                    StopTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);

                    double secondsForEpoch = swEpoch.Elapsed.TotalSeconds;
                    double nbStepsByEpoch = ((double)trainingDataSetCpu.Count) / miniBatchSize;
                    var msByStep = (1000 * secondsForEpoch) / nbStepsByEpoch;
                    Info("Epoch " + epoch + "/" + numEpochs + " - " + Math.Round(secondsForEpoch, 0) + "s " + Math.Round(msByStep, 0) + "ms/step - lr: "+Math.Round(learningRateAtEpochStart, 8)+" - "+lossAndAccuracyMsg);
                    Info(MemoryInfo());
                    //if it is the last epoch, we'll save Layer KPI
                    if (epoch == numEpochs)
                    {
                        LogDebug(LayersKpi());
                    }

                    #region we save stats about the just finished epoch
                    var currentEpochData = new EpochData(epoch, learningRateAtEpochStart, lrMultiplicativeFactorFromReduceLrOnPlateau, trainLossAndAccuracyForEpoch.Item1, trainLossAndAccuracyForEpoch.Item2, validationLossAndAccuracy?.Item1 ?? double.NaN, validationLossAndAccuracy?.Item2 ?? double.NaN, secondsForEpoch);
                    _epochData.Add(currentEpochData);
                    #endregion

                    #region we save the network in a file if necessary
                    if (   //if we have finished training
                           ((epoch == numEpochs) && (numEpochs > 10))
                            //or if we should save the network every 'Config.AutoSaveIntervalInMinuts' minuts
                        || ( (Config.AutoSaveIntervalInMinutes>=0) && (DateTime.Now - lastAutoSaveTime).TotalMinutes > Config.AutoSaveIntervalInMinutes)
                        || learningRateComputer.ShouldCreateSnapshotForEpoch(epoch)
                        )
                    {
                        var swSaveTime = Stopwatch.StartNew();
                        var fileName = Path.Combine(Config.LogDirectory, UniqueId + "_" + epoch + ".txt");
                        Save(fileName);
                        Info("Network '" + Description + "' saved in " + fileName + " in " + Math.Round(swSaveTime.Elapsed.TotalSeconds, 1) + "s");
                        lastAutoSaveTime = DateTime.Now;
                    }
                    #endregion

                    if (Config.SaveNetworkStatsAfterEachEpoch)
                    {
                        var networkStatFileName = Path.Combine(Config.LogDirectory, UniqueId + "_" + epoch + "_NetworkStats.txt");
                        Info("Saving network '" + Description + "' stats in " + networkStatFileName);
                        File.WriteAllText(networkStatFileName, ContentStats());
                    }
                }

                string line = "";
                try
                {
                    //We save the results of the net
                    line = DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                        + Description.Replace(';', '_') + ";"
                        + DeviceName() + ";"
                        + TotalParams + ";"
                        + numEpochs + ";"
                        + miniBatchSize + ";"
                        + learningRateComputer.LearningRate(1, 0, nbBlocksInEpoch, 1.0) + ";"
                        + _spInternalFit.Elapsed.TotalSeconds + ";"
                        + (_spInternalFit.Elapsed.TotalSeconds / numEpochs) + ";"
                        + validationLossAndAccuracy?.Item1 + ";"
                        + validationLossAndAccuracy?.Item2
                        + Environment.NewLine;
                    if (!Config.DisableLogging)
                    {
                        var testsCsv = string.IsNullOrEmpty(trainingDataSetCpu.Name)?"Tests.csv": ("Tests_"+ trainingDataSetCpu.Name + ".csv");
                        File.AppendAllText(Utils.ConcatenatePathWithFileName(Config.LogDirectory, testsCsv), line);
                    }
                }
                catch (Exception e)
                {
                    Info("fail to add line in file:" + Environment.NewLine + line + Environment.NewLine + e);
                    // ignored
                }

                Info("Training '"+ Description+"' for " + numEpochs + " epochs took: " + _spInternalFit.Elapsed.TotalSeconds + "s");
                if (!string.IsNullOrEmpty(Description))
                {
                    LogDebug("Network Name: "+Description);
                }
                _spInternalFit.Stop();
            }
            catch (Exception e)
            {
                Info(e.ToString());
                throw;
            }
        }

        #region compute Loss and Accuracy
        //returns : Tuple<loss, accuracy>
        public Tuple<double, double> ComputeLossAndAccuracyForTestDataSet(int miniBatchSize, IDataSet testDataSet)
        {
            //We perform a mini batch gradient descent in Testing mode:
            //  there will be no shuffling/data augmentation.
            var yPredicted = MiniBatchGradientDescent(testDataSet, miniBatchSize);
            return ComputeLossAndAccuracyForEntireBatch(testDataSet.Y, yPredicted);
        }
        private Tuple<double, double> ComputeLossAndAccuracyForEntireBatch(Tensor yExpected, Tensor yPredicted)
        {
            _swComputeAccuracy?.Start();
            yExpected = ReformatToCorrectDevice_GPU_or_CPU(yExpected);
            yPredicted = ReformatToCorrectDevice_GPU_or_CPU(yPredicted);
            MemoryPool.GetNotInitializedFloatTensor(ref _bufferComputeAccuracy, new[] { yExpected.Shape[0] }, nameof(_bufferComputeAccuracy));
            var accuracy = yExpected.ComputeAccuracy(yPredicted, _bufferComputeAccuracy);
            _swComputeAccuracy?.Stop();
            _swComputeLoss?.Start();
            MemoryPool.GetNotInitializedFloatTensor(ref _bufferComputeLoss, new[] { yExpected.Shape[0] }, nameof(_bufferComputeLoss));
            var totalLoss = yExpected.ComputeLoss(yPredicted, Config.LossFunction, _bufferComputeLoss);
            _swComputeLoss?.Stop();
            return Tuple.Create(totalLoss, accuracy);
        }
        private static string LossAndAccuracyToString(Tuple<double, double> lossAndAccuracy, string prefix)
        {
            return prefix + "loss: " + Math.Round(lossAndAccuracy.Item1, 4) + " - " + prefix + "acc: " + Math.Round(lossAndAccuracy.Item2, 4);
        }
        #endregion

        public void OnLayerAddOrRemove()
        {
            _bytesByBatchSize = null; //we need tor recompute the batch size in bytes
        }

        #region profiling
        private string LayersKpi()
        {
            var totalSeconds = _spInternalFit.Elapsed.TotalSeconds;
            var result = "Took " + Math.Round(totalSeconds, 1) + "s";
            result += " (Loss:" + Math.Round(100 * _swComputeLoss.Elapsed.TotalSeconds / totalSeconds, 0) + "%+Accuracy:"+ Math.Round(100 * _swComputeAccuracy.Elapsed.TotalSeconds / totalSeconds, 0) +"%])"+ Environment.NewLine;
            result += KpiByLayerType(totalSeconds);
            return result;
        }

        private string KpiByLayerType(double totalSeconds)
        {
            double PercentageOfTimeTaken(IDictionary<string, Stopwatch> layerTypeToTimer, string layerType)
            {
                return layerTypeToTimer.TryGetValue(layerType, out var sw) ? (sw.Elapsed.TotalSeconds/Math.Max(totalSeconds, 1e-6)) : 0.0;
            }
            string ParentLayerName(string keyName)
            {
                int idx = keyName.IndexOf(">", StringComparison.Ordinal);
                return (idx > 0) ? keyName.Substring(0, idx) : keyName;
            }
            double ParentTime(string keyName, List<Tuple<string, double, double, double, double>> values)
            {
                var parent = ParentLayerName(keyName);
                var parentTuple = values.FirstOrDefault(t => t.Item1 == parent);
                return parentTuple==null?0:parentTuple.Item2 + parentTuple.Item3 + parentTuple.Item4 + parentTuple.Item5;
            }
            var data = new List<Tuple<string, double, double, double, double>>();
            var separatingLine = new string('=', 100);
            var allKeys = ForwardPropagationTrainingTime.Keys.Union(BackwardPropagationTime.Keys).Union(ForwardPropagationInferenceTime.Keys).Union(_updateWeightsTime.Keys).ToList();
            foreach (var layerType in allKeys)
            {
                data.Add(Tuple.Create(layerType, PercentageOfTimeTaken(ForwardPropagationTrainingTime, layerType), PercentageOfTimeTaken(BackwardPropagationTime, layerType), PercentageOfTimeTaken(ForwardPropagationInferenceTime, layerType), PercentageOfTimeTaken(_updateWeightsTime, layerType)));
            }

            data = data.OrderByDescending(t => ParentTime(t.Item1, data)).ThenBy(t => t.Item1).ToList();
            var result = separatingLine + Environment.NewLine;
            result += "LayerName              Forward(Training)  Backward(Training)  Forward(Inference)        UpdateHeight" + Environment.NewLine;
            result += separatingLine + Environment.NewLine;
            result += string.Join(Environment.NewLine, data.Select(d => KpiByLayerTypeSingleLine(d.Item1, d.Item2, d.Item3, d.Item4, d.Item5))) + Environment.NewLine;
            //we compute the total by column
            result += separatingLine+Environment.NewLine;
            var dataWithoutDuplicate = data.Where(t => !t.Item1.Contains(">")).ToList();
            result += KpiByLayerTypeSingleLine("", dataWithoutDuplicate.Select(t => t.Item2).Sum(), dataWithoutDuplicate.Select(t => t.Item3).Sum(), dataWithoutDuplicate.Select(t => t.Item4).Sum(), dataWithoutDuplicate.Select(t => t.Item5).Sum()) + Environment.NewLine;
            result += separatingLine + Environment.NewLine;
            return result;
        }

        private static string KpiByLayerTypeSingleLine(string layerType, double forwardPropagationTraining, double forwardPropagationInference, double backwardPropagation, double totalUpdateWeights)
        {
            string AsDisplayString(double d)
            {
                return Math.Round(d * 100, 1) + "%";
            }
            string SubCategoryLayerName(string keyName)
            {
                int idx = keyName.IndexOf(">", StringComparison.Ordinal);
                return (idx >= 0) ? keyName.Substring(idx) : "";
            }
            const int columnWidth = 20;
            return (layerType.Contains(">")?$"{SubCategoryLayerName(layerType),20}":$"{layerType,-20}")
                    +$"{AsDisplayString(forwardPropagationTraining),columnWidth}{AsDisplayString(forwardPropagationInference),columnWidth}{AsDisplayString(backwardPropagation),columnWidth}{AsDisplayString(totalUpdateWeights),columnWidth}".TrimEnd();
        }

        public static void StartTimer(string key, IDictionary<string, Stopwatch> layerTypeToStopWatch)
        {
            if (layerTypeToStopWatch.TryGetValue(key, out Stopwatch sw))
            {
                sw.Start();
                return;
            }
            layerTypeToStopWatch[key] = Stopwatch.StartNew();
        }

        public static void StopTimer(string key, IDictionary<string, Stopwatch> layerTypeToStopWatch)
        {
            Debug.Assert(layerTypeToStopWatch.ContainsKey(key));
            layerTypeToStopWatch[key].Stop();
        }
        #endregion

        public void LoadFromH5Dataset(List<Tuple<string, Tensor>> h5FileDatasetAsList, NetworkConfig.CompatibilityModeEnum originFramework)
        {
            var h5FileDataset = new Dictionary<string, Tensor>();
            foreach (var e in h5FileDatasetAsList)
            {
                if (h5FileDataset.ContainsKey(e.Item1))
                {
                    throw new ArgumentException("duplicate dataset path for "+e.Item1);
                }
                h5FileDataset[e.Item1] = e.Item2;
            }

            foreach (var l in Layers.Skip(1)) //we skip the input layer
            {
                l.LoadFromH5Dataset(h5FileDataset, originFramework);
            }
        }
        // ReSharper disable once UnusedMember.Global
        public List<Tuple<string, Tensor>> SaveToH5Dataset(NetworkConfig.CompatibilityModeEnum targetFramework)
        {
            var result = new List<Tuple<string, Tensor>>();
            foreach (var l in Layers.Skip(1)) //we skip the input layer
            {
                l.SaveToH5Dataset(result, targetFramework);
            }
            return result;
        }
        public void Info(string msg) { Config.Logger.Info(msg); }
        public void LogDebug(string msg) { Config.Logger.Debug(msg); }

        /// <summary>
        /// = ForwardPropagation
        /// this method is only used for tests
        /// </summary>
        /// <param name="X"></param>
        /// <param name="isTraining">
        /// true if we are training the network (the goal is to update weights)
        /// false for inference only (we'll use existing weights to make a prediction)
        /// </param>
        /// <returns></returns>
        public Tensor Predict(Tensor X, bool isTraining)
        {
            X = ReformatToCorrectDevice_GPU_or_CPU(X);
            var yPredicted = MemoryPool.GetNotInitializedFloatTensor(Layers.Last().OutputShape(X.Shape[0]), "predictionBuffer");
            X = ReformatToCorrectDevice_GPU_or_CPU(X);
            PropagationManager.Forward(X, isTraining, yPredicted);
            return yPredicted;
        }

        /// <summary>
        /// Perform a mini batch gradient descent for an entire epoch, each mini batch will have 'miniBatchSize' elements
        /// </summary>
        /// <param name="dataSet">Expected Input and output (= dataSet.Y) </param>
        /// <param name="miniBatchSize"></param>
        /// <param name="learningRateComputerIfTraining">
        /// null if we are just using the network to predict the results (without updating weights)
        /// not null if we need to update the weights at the end of each mini batch</param>
        /// <param name="callBackToStop">Optional callback to be called at the end of each mini batch,
        /// parameters are: 'mini batch expected output' + 'mini batch observed output' + 'current block Id'
        /// If the callback returns true we should stop the computation</param>
        /// <returns>observed output associated with the input 'x'</returns>
        public Tensor MiniBatchGradientDescent(IDataSet dataSet, int miniBatchSize = -1, ILearningRateComputer learningRateComputerIfTraining = null, Func<Tensor, Tensor, int, int, int, bool> callBackToStop = null)
        {
            //last time we display a progress on the screen for the current min batch descent
            var miniBatchGradientDescentStart = DateTime.Now;
            var lastStatsUpdate = miniBatchGradientDescentStart;

            bool isTraining = learningRateComputerIfTraining != null;
            var entireBatchSize = dataSet.Count;
            if (miniBatchSize <= 0)
            {
                miniBatchSize = MaxMiniBatchSize(dataSet.XMiniBatch_Shape(1), isTraining);
            }

            //number of mini batch block in current epoch
            int nbMiniBatchBlock = NbBlocksInEpoch(miniBatchSize, entireBatchSize);
            //the first epoch is #1
            int epoch = _epochData.Count + 1;
            var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputerIfTraining?.MultiplicativeFactorFromReduceLrOnPlateau(_epochData) ?? 1.0;
            MemoryPool.GetNotInitializedFloatTensor(ref _yPredictedBufferForEntireBatch, dataSet.Y_Shape);
            MemoryPool.GetNotInitializedFloatTensor(ref _yExpectedBufferForEntireBatch, dataSet.Y_Shape);
            var xMiniBatch = MemoryPool.GetNotInitializedFloatTensor(dataSet.XMiniBatch_Shape(miniBatchSize), "xMiniBatch");
            _xMiniBatchCpu.Reshape(xMiniBatch.Shape);
            _yExpectedMiniBatchCpu.Reshape(dataSet.YMiniBatch_Shape(miniBatchSize));

            var shuffledElementId = Enumerable.Range(0, dataSet.Count).ToArray();
            if (epoch >= 2 && Config.RandomizeOrder && isTraining)
            {
                Utils.Shuffle(shuffledElementId, Config.Rand);
            }

            int blockId = 0;
            int nbProcessed = 0;
            while(nbProcessed < entireBatchSize)
            {
                var blockSize = Math.Min(entireBatchSize- nbProcessed, miniBatchSize);
                MemoryPool.GetNotInitializedFloatTensor(ref xMiniBatch, dataSet.XMiniBatch_Shape(blockSize), nameof(xMiniBatch));
                var yExpectedMiniBatch = _yExpectedBufferForEntireBatch.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                _xMiniBatchCpu.Reshape(xMiniBatch.Shape);
                _yExpectedMiniBatchCpu.Reshape(yExpectedMiniBatch.Shape);
                StartTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                dataSet.LoadMiniBatch(epoch, isTraining, shuffledElementId, blockId * miniBatchSize, Config.DataAugmentation, _xMiniBatchCpu, _yExpectedMiniBatchCpu);
                StopTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);

                //we copy mini batch content (_xMiniBatchCpu) from CPU to appropriate target (CPU or GPU)
                _xMiniBatchCpu.CopyTo(xMiniBatch);
                _yExpectedMiniBatchCpu.CopyTo(yExpectedMiniBatch);

                var yPredictedMiniBatch = _yPredictedBufferForEntireBatch.ExtractSubTensor(blockId * miniBatchSize, blockSize);
                PropagationManager.Forward(xMiniBatch, isTraining, yPredictedMiniBatch);
                if (isTraining)
                {
                    PropagationManager.Backward(yExpectedMiniBatch, yPredictedMiniBatch);
                    PropagationManager.UpdateWeights(blockSize, learningRateComputerIfTraining.LearningRate(epoch, blockId, nbMiniBatchBlock, lrMultiplicativeFactorFromReduceLrOnPlateau));
                }
                nbProcessed += blockSize;
                if (callBackToStop != null && callBackToStop(yExpectedMiniBatch, yPredictedMiniBatch, blockId, nbMiniBatchBlock, epoch))
                {
                    break;
                }
                ++blockId;

                if ((DateTime.Now-lastStatsUpdate).TotalSeconds> 5*60)
                {
                    var percentageDoneInEpoch = ((double) nbProcessed) / entireBatchSize;
                    var secondsSinceStartOfEpoch = (DateTime.Now - miniBatchGradientDescentStart).TotalSeconds;
                    var expectedSecondsToPerformEntireEpoch = secondsSinceStartOfEpoch / percentageDoneInEpoch;
                    Info("Epoch " + epoch + " in progress: " + Math.Round(100.0* percentageDoneInEpoch, 1) + "% performed ("+ Math.Round(secondsSinceStartOfEpoch, 0) + "s/"+Math.Round(expectedSecondsToPerformEntireEpoch,0)+"s)");
                    LogDebug(MemoryInfo());
                    lastStatsUpdate = DateTime.Now;
                }
            }

            MemoryPool.FreeMemory(ref xMiniBatch);
            return _yPredictedBufferForEntireBatch;
        }
        public int LastLayerIndex => Layers.Last().LayerIndex;

        private bool UseGPU => GpuDeviceId >= 0;
        private string MemoryInfo()
        {
            string result = "Private Memory: " + Utils.MemoryBytesToString((ulong)Process.GetCurrentProcess().PrivateMemorySize64);
            result += " - Managed Memory: " + Utils.MemoryBytesToString((ulong)GC.GetTotalMemory(false));
            result += " - " + MemoryPool.MemoryInfo();
            if (UseGPU)
            {
                result += " - " + GpuWrapper.MemoryInfo();
            }
            result += " - CurrentThreadId#" + System.Threading.Thread.CurrentThread.ManagedThreadId;
            return result;
        }
        private static int NbBlocksInEpoch(int miniBatchSize, int entireBatchSize)
        {
            return (entireBatchSize + miniBatchSize - 1) / miniBatchSize;
        }
        private string UniqueId => (string.IsNullOrEmpty(Description) ? "Network" : Utils.ToValidFileName(Description)) + "_" + _timeStampCreation.ToString("yyyyMMdd_HHmm", CultureInfo.InvariantCulture);
        private void CreateLogDirectoryIfNeeded()
        {
            if (!string.IsNullOrEmpty(Config.LogDirectory) && !Directory.Exists(Config.LogDirectory))
            {
                Directory.CreateDirectory(Config.LogDirectory);
            }
        }
        /// <summary>
        /// bytes / batchSize needed for forward & backward propagation
        /// </summary>
        /// <param name="xShape"></param>
        /// <param name="isTraining"></param>
        private ulong BytesByBatchSizeForwardAndBackward(int[] xShape, bool isTraining)
        {
            if (_bytesByBatchSize.HasValue)
            {
                return _bytesByBatchSize.Value;
            }
            Debug.Assert(xShape.Length == 4);   // tensor of shape (n, c, h, w)
            Debug.Assert(xShape[0] == 1);       //batch size must be  1
            using var mockMemoryPooling = new TensorMemoryPool(null, true);
            using var propagationManager = new PropagationManager(Layers, mockMemoryPooling, ForwardPropagationTrainingTime, ForwardPropagationInferenceTime, BackwardPropagationTime, _updateWeightsTime);
            var x = mockMemoryPooling.GetNotInitializedFloatTensor(xShape, "x");
            var yPredicted = mockMemoryPooling.GetNotInitializedFloatTensor(Layers.Last().OutputShape(x.Shape[0]), "yPredicted");
            propagationManager.Forward(x, isTraining, yPredicted);
            if (isTraining)
            {
                var yExpected = yPredicted;
                propagationManager.Backward(yExpected, yPredicted);
            }
            return mockMemoryPooling.CapacityInBytes;
        }
        private ulong BytesIndependentOfBatchSize => Layers.Select(x => x.BytesIndependentOfBatchSize).Sum();
        private Tensor ReformatToCorrectDevice_GPU_or_CPU(Tensor X)
        {
            if (X == null || UseGPU == X.UseGPU)
            {
                return X;
            }
            if (X.UseGPU)
            {
                throw new NotImplementedException("can not reformat type that are stored in GPU");
            }
            return X.ToGPU<float>(GpuWrapper);
        }
        public string Summary()
        {
            return Layers.Any(x => x.PreviousLayers.Count >= 2) ? SummaryWithConnectedTo() : SummaryWithoutConnectedTo();
        }
        private string SummaryWithConnectedTo()
        {
            const int firstColumnWidth = 32;
            const int secondColumnWidth = 21;
            const int thirdColumnWidth = 12;
            const int forthColumnWidth = 33;
            var line0 = new string('_', firstColumnWidth + secondColumnWidth + thirdColumnWidth + forthColumnWidth);
            var line1 = new string('=', line0.Length);
            string result = "";
            if (!string.IsNullOrEmpty(Description))
            {
                result += "Network Name: " + Description + Environment.NewLine;
            }
            result += line0 + Environment.NewLine;
            result += "Layer (type)                    Output Shape         Param #     Connected to" + Environment.NewLine;
            result += line1 + Environment.NewLine;
            foreach (var layer in Layers)
            {
                var outputShape = Utils.ShapeToStringWithBatchSize(layer.OutputShape(1));
                var firstColumn = layer.LayerName + " (" + layer.Type() + ")";
                if (firstColumn.Length > firstColumnWidth - 1)
                {
                    firstColumn = firstColumn.Substring(0, firstColumnWidth - 1);
                }
                var previousLayers = layer.PreviousLayers.OrderBy(x => x.LayerIndex).ToList();
                var firstPreviousLayer = (previousLayers.Count == 0 ? "" : previousLayers[0].LayerName + "[0][0]");
                result += ($"{firstColumn,-firstColumnWidth}{outputShape,-secondColumnWidth}{layer.TotalParams,-thirdColumnWidth}{firstPreviousLayer,-forthColumnWidth}").TrimEnd() + Environment.NewLine;
                for (int i = 1; i < previousLayers.Count; ++i)
                {
                    result += ($"{"",-(firstColumnWidth + secondColumnWidth + thirdColumnWidth)}{previousLayers[i].LayerName + "[0][0]",-forthColumnWidth}").TrimEnd() + Environment.NewLine;
                }
                result += (layer.IsOutputLayer ? line1 : line0) + Environment.NewLine;
            }
            result += "Total params: " + TotalParams;
            return result;
        }
        private string SummaryWithoutConnectedTo()
        {
            const int firstColumnWidth = 29;
            const int secondColumnWidth = 26;
            const int thirdColumnWidth = 10;
            var line0 = new string('_', firstColumnWidth + secondColumnWidth + thirdColumnWidth);
            var line1 = new string('=', line0.Length);
            string result = "";
            if (!string.IsNullOrEmpty(Description))
            {
                result += "Network Name: " + Description + Environment.NewLine;
            }
            result += line0 + Environment.NewLine;
            result += "Layer (Type)                 Output Shape              Param #" + Environment.NewLine;
            result += line1 + Environment.NewLine;
            foreach (var l in Layers)
            {
                var outputShape = Utils.ShapeToStringWithBatchSize(l.OutputShape(1));
                var firstColumn = l.LayerName + " (" + l.Type() + ")";
                if (firstColumn.Length > firstColumnWidth - 1)
                {
                    firstColumn = firstColumn.Substring(0, firstColumnWidth - 1);
                }
                result += ($"{firstColumn,-firstColumnWidth}{outputShape,-secondColumnWidth}{l.TotalParams,-thirdColumnWidth}").TrimEnd() + Environment.NewLine;
                result += (l.IsOutputLayer ? line1 : line0) + Environment.NewLine;
            }
            result += "Total params: " + TotalParams;
            return result;
        }
        /// <summary>
        /// return the index of the layer whose name is 'layerName' or -1 if there is no such layer
        /// </summary>
        /// <param name="layerName">the layer name for which we want to know the layer index</param>
        /// <returns></returns>
        private int LayerNameToLayerIndex(string layerName)
        {
            for (var layerIndex = 0; layerIndex < Layers.Count; layerIndex++)
            {
                var layer = Layers[layerIndex];
                if (string.Equals(layer.LayerName, layerName ?? "", StringComparison.OrdinalIgnoreCase))
                {
                    return layerIndex;
                }
            }
            return -1;
        }
        private void ResetWeights()
        {
            foreach (var l in Layers)
            {
                if (l != null && l.Trainable)
                {
                    l.ResetWeights();
                }
            }
        }
        private int MaxMiniBatchSize(int[] xShape, bool isTraining)
        {
            var freeMemoryInBytes = UseGPU ? (ulong)GpuWrapper.AvailableGpuMemoryInBytes() : Utils.AvailableRamMemoryInBytes();
            var bytesByBatchSizeForwardAndBackward = BytesByBatchSizeForwardAndBackward(xShape, isTraining);
            int maxMiniBatchSize = MaxMiniBatchSize(bytesByBatchSizeForwardAndBackward, BytesIndependentOfBatchSize, freeMemoryInBytes);
            LogDebug("Max MiniBatchSize=" + maxMiniBatchSize + " (free memory=" + Utils.MemoryBytesToString(freeMemoryInBytes) + ")");
            return maxMiniBatchSize;
        }
        //TODO add tests
        private static int MaxMiniBatchSize(ulong bytesByBatchSize, ulong bytesIndependentOfBatchSize, ulong freeMemoryInBytes)
        {
            freeMemoryInBytes -= bytesIndependentOfBatchSize;

            freeMemoryInBytes -= 1_200_000_000;

            //freeMemoryInBytes = (80* freeMemoryInBytes)/100;
            //freeMemoryInBytes = (85 * freeMemoryInBytes) / 100;

            ulong miniBatchSize = freeMemoryInBytes / bytesByBatchSize;
            if (miniBatchSize > 4)
            {
                miniBatchSize -= (miniBatchSize % 4);
            }

            /*
            ulong miniBatchSize = 1;
            while ( (2UL * miniBatchSize * bytesByBatchSize) < freeMemoryInBytes)
            {
                miniBatchSize *= 2;
            }
            */
            return (int)miniBatchSize;
        }
        private string ContentStats()
        {
            var sb = new StringBuilder();
            foreach (var l in Layers)
            {
                sb.Append(new string('-', 80) + Environment.NewLine);
                sb.Append("Layer:" + l.LayerName + Environment.NewLine);
                var contentStats = l.ContentStats();
                if (!string.IsNullOrEmpty(contentStats))
                {
                    sb.Append(contentStats + Environment.NewLine);
                }
            }
            return sb.ToString();
        }
    }
}
