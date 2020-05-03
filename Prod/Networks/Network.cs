﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Text;
using System.Threading;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Optimizers;
using H5GroupId = System.Int64;


namespace SharpNet.Networks
{
    public partial class Network : IDisposable
    {
        #region private fields
        private readonly List<EpochData> _epochData = new List<EpochData>();
        /// <summary>
        /// all resources (CPU or GPU) available for the current network
        /// values superior or equal to 0 means GPU resources (device)
        /// values strictly less then 0 mean CPU resources (host)
        /// </summary>
        private readonly List<int> _resourceIds;
        private Tensor _bufferComputeAccuracy;
        private Tensor _bufferComputeLoss;
        private readonly DateTime _timeStampCreation = DateTime.Now;
        // bytes/batchSize needed for forward & backward propagation
        private ulong? _bytesByBatchSize;
        #endregion

        #region public fields
        public NetworkConfig Config { get; }
        public List<Layer> Layers { get; } = new List<Layer>();
        public string Description { private get; set; } = "";
        public PropagationManager PropagationManager { get; }
        public TensorMemoryPool MemoryPool { get; }
        public GPUWrapper GpuWrapper { get; }
        #endregion

        /// <param name="config"></param>
        /// <param name="resourceIds">
        ///     list of resources available for the network
        ///     if gpuDeviceIds.Count == 1
        ///     if masterNetworkIfAny == null:
        ///     all computation will be done in a single network (no parallel computing between different networks
        ///     else:
        ///     we are using several networks in // for doing the computation
        ///     and we are in a slave network doing part of this parallel computation for the master network 'masterNetworkIfAny'
        ///     else: (gpuDeviceIds.Count >= 2)
        ///     we are using several networks in // for doing the computation
        ///     and we are in the master network doing part of this parallel computation
        ///     this master network will use resourceId gpuDeviceIds[1]
        ///     slaves network will use resourceId gpuDeviceIds[1:]
        /// 
        ///     for each resourceId in this list:
        ///     if resourceId strictly less then 0:
        ///     use CPU resource (no GPU usage)
        ///     else:
        ///     run the network on the GPU with device Id = resourceId
        /// </param>
        /// <param name="masterNetworkIfAny">
        ///     if the current network is a slave network doing computation for its master network:
        ///     the reference of the master network
        ///     else:
        ///     null
        /// </param>
        public Network(NetworkConfig config, List<int> resourceIds, Network masterNetworkIfAny = null)
        {
            //a slave network will have access to only one resource (Cpu or GPU device id)
            Debug.Assert(masterNetworkIfAny == null || resourceIds.Count == 1);
            Config = config;
            _masterNetworkIfAny = masterNetworkIfAny;
            _resourceIds = resourceIds.ToList();
            GpuWrapper = UseGPU ? GPUWrapper.FromDeviceId(_resourceIds[0]) : null;
            _swComputeLoss = new Stopwatch();
            _swComputeAccuracy = new Stopwatch();
            CreateLogDirectoryIfNeeded();
            MemoryPool = new TensorMemoryPool(GpuWrapper, false);
            PropagationManager = new PropagationManager(Layers, MemoryPool, ForwardPropagationTrainingTime, ForwardPropagationInferenceTime, BackwardPropagationTime, _updateWeightsTime);
            if (IsMaster && resourceIds.Count>=2)
            {
                //we create the slave networks
                foreach(var slaveResourceId in resourceIds.Skip(1))
                {
                    new Thread(() => SlaveThread(this, slaveResourceId)).Start();
                }
                while (_slaveNetworks.Count != resourceIds.Count - 1)
                {
                    Thread.Sleep(1);
                }
            }
        }


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
                var lambdaL2Regularization = denseLayer.LambdaL2Regularization;
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
        
        public string DeviceName() { return GpuWrapper?.DeviceName(); }

        public void Dispose()
        {
            if (IsMaster)
            {
                LogDebug("Before clearing memory: " + MemoryInfo());
                GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
                GC.Collect();
                SetStatusForAllSlaves(SLAVE_NETWORK_STATUS.TO_ABORT);
            }
            Layers.ForEach(l => l?.Dispose());
            Layers.Clear();
            PropagationManager.Dispose();
            _epochData.Clear();
            MemoryPool.FreeFloatTensor(ref _bufferComputeAccuracy);
            MemoryPool.FreeFloatTensor(ref _bufferComputeLoss);
            MemoryPool.FreeFloatTensor(ref _yPredictedForEpoch);
            MemoryPool.FreeFloatTensor(ref _yExpectedForEpoch);
            MemoryPool.FreeFloatTensor(ref _x_miniBatch);
            MemoryPool.FreeFloatTensor(ref _bufferAddGradientFromSlaveNetwork);
            MemoryPool.FreeFloatTensor(ref _yExpected_miniBatch_slave);
            MemoryPool.FreeFloatTensor(ref _yPredicted_miniBatch_slave);
            MemoryPool.FreeFloatTensor(ref _compactedParametersIfAny);
            MemoryPool.FreeFloatTensor(ref _compactedGradientsIfAny);
            GpuWrapper?.Reset();
            MemoryPool.Dispose();

            if (IsMaster)
            {
                WaitForAllSlavesInStatus(SLAVE_NETWORK_STATUS.DISPOSED);
                _slaveNetworks.Clear();
                GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
                GC.Collect();
                LogDebug("After clearing memory: " + MemoryInfo());
            }

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
        public Network MultiplyLayer(int previousLayerIndex1, int previousLayerIndexDiagonalMatrix, string layerName = "")
        {
            Layers.Add(new MultiplyLayer(previousLayerIndex1, previousLayerIndexDiagonalMatrix, this, layerName));
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
            result += Utils.MemoryBytesToString(BytesByBatchSizeForwardAndBackward(xShape, true)) + "/batchSize";
            return result;
        }
        public Optimizer GetOptimizer(int[] weightShape, int[] biasShape)
        {
            switch (Config.OptimizerType)
            {
                case Optimizer.OptimizationEnum.Adam: return new Adam(MemoryPool, Config.Adam_beta1, Config.Adam_beta2, Config.Adam_epsilon, weightShape, biasShape);
                case Optimizer.OptimizationEnum.SGD: return new Sgd(MemoryPool, Config.SGD_momentum, Config.SGD_usenesterov, weightShape, biasShape);
                default: return VanillaSgd.Instance;
            }
        }
        public int TotalParams => Layers.SelectMany(x => x.Parameters).Select(t=> t.Item1.Count).Sum();

        #region serialization
        // ReSharper disable once UnusedMember.Global
        public static Network ValueOf(string modelFilePath, int?overrideGpuDeviceId = null)
        {
            //we load the model (network description)
            var allLines = File.ReadAllLines(modelFilePath);
            var dicoFirstLine = Serializer.Deserialize(allLines[0]);
            var config = NetworkConfig.ValueOf(dicoFirstLine);
            int[] gpuDeviceId = overrideGpuDeviceId.HasValue ? new []{overrideGpuDeviceId.Value} : (int[])dicoFirstLine[nameof(_resourceIds)];
            var network = new Network(config, gpuDeviceId.ToList());
            var epochsData = (EpochData[])dicoFirstLine[nameof(_epochData)];
            network._epochData.AddRange(epochsData);
            network.Description = dicoFirstLine.TryGet<string>(nameof(Description))??"";
            for (int i = 1; i < allLines.Length; ++i)
            {
                network.Layers.Add(Layer.ValueOf(Serializer.Deserialize(allLines[i]), network));
            }

            //we load the parameters into the network
            var parametersFilePath = Utils.UpdateFilePathChangingExtension(modelFilePath, "", "", ".h5");
            if (File.Exists(parametersFilePath))
            {
                network.LoadParametersFromH5File(parametersFilePath, network.Config.CompatibilityMode);
            }

            return network;
        }

        private void Save(string modelFilePath)
        {
            //we save the model
            var firstLine = new Serializer()
                .Add(nameof(Description), Description)
                .Add(Config.Serialize())
                .Add(nameof(_resourceIds), _resourceIds.ToArray())
                .Add(nameof(_epochData), _epochData.ToArray())
                .ToString();
            File.AppendAllLines(modelFilePath, new[] { firstLine });
            foreach (var l in Layers)
            {
                File.AppendAllLines(modelFilePath, new[] { l.Serialize() });
            }


            SaveParametersToH5File(Utils.UpdateFilePathChangingExtension(modelFilePath, "", "", ".h5"));
        }

        /// <summary>
        /// load the parameters from h5 file 'h5FilePath' into the network
        /// </summary>
        /// <param name="h5FilePath"></param>
        /// <param name="originFramework"></param>
        private void SaveParametersToH5File(string h5FilePath)
        {
            if (File.Exists(h5FilePath))
            {
                File.Delete(h5FilePath);
            }
            using var h5File = new H5File(h5FilePath);
            foreach (var l in Layers)
            {
                foreach (var p in l.GetParametersAsCpuFloatTensors(Config.CompatibilityMode))
                {
                    h5File.Write(p.Key, p.Value);
                }
            }
        }

        /// <summary>
        /// load the parameters from h5 file 'h5FilePath' into the network
        /// </summary>
        /// <param name="h5FilePath"></param>
        /// <param name="originFramework"></param>
        public void LoadParametersFromH5File(string h5FilePath, NetworkConfig.CompatibilityModeEnum originFramework)
        {
            using var h5File = new H5File(h5FilePath);
            var h5FileParameters = h5File.Datasets();
            Layers.ForEach(l=>l.LoadParameters(h5FileParameters, originFramework));

            var networkParametersKeys = Layers.SelectMany(t => t.Parameters).Select(t => t.Item2).ToList();
            var elementMissingInH5Files = networkParametersKeys.Except(h5FileParameters.Keys).ToList();
            if (elementMissingInH5Files.Count != 0)
            {
                Info(elementMissingInH5Files.Count + " parameters are missing in file " + h5FilePath + ": " + string.Join(", ", elementMissingInH5Files));
            }
            var elementMissingInNetwork = h5FileParameters.Keys.Except(networkParametersKeys).ToList();
            if (elementMissingInNetwork.Count != 0)
            {
                Info(elementMissingInNetwork.Count + " parameters are missing in network " + h5FilePath + ": " + string.Join(", ", elementMissingInNetwork));
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
            if (IsMaster)
            {
                _slaveNetworks.ForEach(n=>n.LogContent());
            }
        }
        #endregion

        public double FindBestLearningRate(IDataSet trainingDataSet, double minLearningRate, double maxLearningRate, int miniBatchSizeForAllWorkers = -1)
        {
            Info("Looking for best learning rate...");
            ResetWeights(); //restore weights to there original values
            var maxMiniBatchSizeForAllWorkers = MaxMiniBatchSizeForAllWorkers(trainingDataSet.XMiniBatch_Shape(1), false);
            if (miniBatchSizeForAllWorkers < 1)
            {
                miniBatchSizeForAllWorkers = maxMiniBatchSizeForAllWorkers;
            }
            else
            {
                if (miniBatchSizeForAllWorkers > maxMiniBatchSizeForAllWorkers)
                {
                    Info("Reducing BatchSize from "+miniBatchSizeForAllWorkers+" to "+ maxMiniBatchSizeForAllWorkers);
                    miniBatchSizeForAllWorkers = maxMiniBatchSizeForAllWorkers;
                }
            }
            var learningRateFinder = new LearningRateFinder(miniBatchSizeForAllWorkers, trainingDataSet.Count, minLearningRate, maxLearningRate);
                

            void CallBackAfterEachMiniBatch(Tensor yExpectedMiniBatch, Tensor yPredictedMiniBatch)
            {
                MemoryPool.GetFloatTensor(ref _bufferComputeLoss, new[] { yExpectedMiniBatch.Shape[0] });
                var blockLoss = yExpectedMiniBatch.ComputeLoss(yPredictedMiniBatch, Config.LossFunction, _bufferComputeLoss);
                learningRateFinder.AddLossForLastBlockId(blockLoss);
            }
            MiniBatchGradientDescentForSingleEpoch(trainingDataSet, miniBatchSizeForAllWorkers, learningRateFinder, (yExpectedMiniBatch, yPredictedMiniBatch) => CallBackAfterEachMiniBatch(yExpectedMiniBatch, yPredictedMiniBatch));
            var fileName = Path.Combine(Config.LogDirectory, UniqueId + "_LearningRateFinder.csv");
            File.WriteAllText(fileName, learningRateFinder.AsCsv());
            Info("Stats stored in: " + fileName);
            var bestLearningRate = learningRateFinder.BestLearningRate();
            Info("Best learning rate: "+ bestLearningRate+ " (with batch size="+miniBatchSizeForAllWorkers+")");
            ResetWeights(); //restore weights to there original values
            return bestLearningRate;
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
                    layer.ResetParameters();
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
        /// <param name="preferredMiniBatchSizeForAllWorkers"></param>
        /// <param name="testDataSetCpuIfAny"></param>
        public void Fit(IDataSet trainingDataSetCpu, ILearningRateComputer learningRateComputer, int numEpochs, int preferredMiniBatchSizeForAllWorkers, IDataSet testDataSetCpuIfAny)
        {
            try
            {
                Debug.Assert(Config.TypeSize == trainingDataSetCpu.TypeSize);
                Debug.Assert(learningRateComputer != null);
                _spInternalFit.Start();
                StartTimer("Fit_Prepare", ForwardPropagationTrainingTime);

                FreezeSelectedLayers();
                
                Info(ToString());
                var maxMiniBatchSizeForAllWorkers = MaxMiniBatchSizeForAllWorkers(trainingDataSetCpu.XMiniBatch_Shape(1), true);
                var miniBatchSizeForAllWorkers = preferredMiniBatchSizeForAllWorkers;
                if (miniBatchSizeForAllWorkers < 1)
                {
                    miniBatchSizeForAllWorkers = maxMiniBatchSizeForAllWorkers;
                    Info("Using (auto) MiniBatchSize of " + miniBatchSizeForAllWorkers);
                }
                else if (miniBatchSizeForAllWorkers > maxMiniBatchSizeForAllWorkers)
                {
                    Info("Reducing MiniBatchSize from "+ miniBatchSizeForAllWorkers+" to "+ maxMiniBatchSizeForAllWorkers+" because of memory limit.");
                    miniBatchSizeForAllWorkers = maxMiniBatchSizeForAllWorkers;
                }

                if (UseGPU)
                {
                    LogDebug(GpuWrapper.ToString());
                }
                LogDebug("Training Set: " + trainingDataSetCpu);
                if (testDataSetCpuIfAny != null)
                {
                    LogDebug("Test Set: " + testDataSetCpuIfAny);
                }
                Info("#Epochs=" + numEpochs + " BatchSize=" + miniBatchSizeForAllWorkers+" Name="+Description);
                if (Config.DisplayTensorContentStats)
                {
                    LogDebug("Initial Tensor Content stats" + Environment.NewLine + ContentStats() + Environment.NewLine);
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
                    var learningRateAtEpochStart = learningRateComputer.LearningRate(epoch, 0, lrMultiplicativeFactorFromReduceLrOnPlateau);
                    var yPredicted = MiniBatchGradientDescentForSingleEpoch(trainingDataSetCpu, miniBatchSizeForAllWorkers, learningRateComputer, null);
                    #endregion

                    //We display stats about the just finished epoch
                    if (Config.DisplayTensorContentStats)
                    {
                        LogDebug("End of Epoch:" + epoch + " Tensor Content stats" + Environment.NewLine+ContentStats()+Environment.NewLine);
                    }

                    StartTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);
                    var trainLossAndAccuracyForEpoch = ComputeLossAndAccuracyForEntireBatch(_yExpectedForEpoch, yPredicted);
                    var lossAndAccuracyMsg = LossAndAccuracyToString(trainLossAndAccuracyForEpoch, "");
                    if (testDataSetCpuIfAny != null)
                    {
                        //We compute the validation (= test) loss&accuracy
                        validationLossAndAccuracy = ComputeLossAndAccuracyForTestDataSet(miniBatchSizeForAllWorkers, testDataSetCpuIfAny);
                        lossAndAccuracyMsg += " - "+LossAndAccuracyToString(validationLossAndAccuracy, "val_");
                    }
                    StopTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);

                    double secondsForEpoch = swEpoch.Elapsed.TotalSeconds;
                    double nbStepsByEpoch = ((double)trainingDataSetCpu.Count) / miniBatchSizeForAllWorkers;
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
                        var modelFilePath = Path.Combine(Config.LogDirectory, UniqueId + "_" + epoch + ".txt");
                        Save(modelFilePath);
                        Info("Network '" + Description + "' saved in " + modelFilePath + " in " + Math.Round(swSaveTime.Elapsed.TotalSeconds, 1) + "s");
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
                        + miniBatchSizeForAllWorkers + ";"
                        + learningRateComputer.LearningRate(1, 0, 1.0) + ";"
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
            var yPredicted = MiniBatchGradientDescentForSingleEpoch(testDataSet, miniBatchSize);
            return ComputeLossAndAccuracyForEntireBatch(testDataSet.Y, yPredicted);
        }
        private Tuple<double, double> ComputeLossAndAccuracyForEntireBatch(Tensor yExpected, Tensor yPredicted)
        {
            _swComputeAccuracy?.Start();
            yExpected = ReformatToCorrectDevice_GPU_or_CPU(yExpected);
            yPredicted = ReformatToCorrectDevice_GPU_or_CPU(yPredicted);
            MemoryPool.GetFloatTensor(ref _bufferComputeAccuracy, new[] { yExpected.Shape[0] });
            var accuracy = yExpected.ComputeAccuracy(yPredicted, _bufferComputeAccuracy);
            _swComputeAccuracy?.Stop();
            _swComputeLoss?.Start();
            MemoryPool.GetFloatTensor(ref _bufferComputeLoss, new[] { yExpected.Shape[0] });
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
            if (_compactedParametersIfAny != null)
            {
                throw new InvalidOperationException("_compactedParametersIfAny is not null");
            }
            if (_compactedGradientsIfAny != null)
            {
                throw new InvalidOperationException("_compactedGradientsIfAny is not null");
            }

            _bytesByBatchSize = null; //we need tor recompute the batch size in bytes
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
            var yPredicted = MemoryPool.GetFloatTensor(Layers.Last().OutputShape(X.Shape[0]));
            X = ReformatToCorrectDevice_GPU_or_CPU(X);
            PropagationManager.Forward(X, yPredicted, isTraining);
            return yPredicted;
        }
        /// <summary>
        /// Perform a mini batch gradient descent for an entire epoch, each mini batch will have (at most) 'miniBatchSize' elements
        /// </summary>
        /// <param name="dataSet">Expected Input and output (= dataSet.Y) </param>
        /// <param name="miniBatchSizeForAllWorkers">
        /// the total size of each miniBatch block.
        /// if several workers are used, this size will be splitted between the workers.
        /// Example:
        ///     if miniBatchSizeForAllWorkers == 512 and 4 workers:
        ///         each worker will compute 128 elements in its miniBatch
        /// </param>
        /// <param name="learningRateComputerIfTraining">
        /// null if we are just using the network to predict the results (without updating weights)
        /// not null if we need to update the weights at the end of each mini batch</param>
        /// <param name="CallBackAfterEachMiniBatch">Optional callback to be called at the end of each mini batch,
        /// parameters are: 'mini batch expected output' + 'mini batch observed output' + 'current block Id'
        /// If the callback returns true we should stop the computation</param>
        /// <returns>observed output associated with the input 'x'</returns>
        public Tensor MiniBatchGradientDescentForSingleEpoch(IDataSet dataSet, int miniBatchSizeForAllWorkers = -1, ILearningRateComputer learningRateComputerIfTraining = null, Action<Tensor, Tensor> CallBackAfterEachMiniBatch = null)
        {
            Debug.Assert(IsMaster);

            if (_slaveNetworks.Any())
            {
                CompactParameters();
                CompactGradients();
            }

            //last time we display a progress on the screen for the current min batch descent
            var miniBatchGradientDescentStart = DateTime.Now;
            var lastStatsUpdate = miniBatchGradientDescentStart;
            bool isTraining = learningRateComputerIfTraining != null;
            //total number of elements to process: they will be processed by block of 'miniBatchSize' elements
            var totalElementCount = dataSet.Count;
            if (miniBatchSizeForAllWorkers <= 0)
            {
                miniBatchSizeForAllWorkers = MaxMiniBatchSizeForAllWorkers(dataSet.XMiniBatch_Shape(1), isTraining);
            }

            //the mini batch size must be a multiple of the number of workers
            Debug.Assert(miniBatchSizeForAllWorkers< DegreeOfParallelism || miniBatchSizeForAllWorkers % DegreeOfParallelism == 0);
            int miniBatchSizeForEachWorker = Math.Max(1, miniBatchSizeForAllWorkers / DegreeOfParallelism);


            //the first epoch is #1
            int epoch = _epochData.Count + 1;
            var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputerIfTraining?.MultiplicativeFactorFromReduceLrOnPlateau(_epochData) ?? 1.0;
            MemoryPool.GetFloatTensor(ref _yPredictedForEpoch, dataSet.Y_Shape);
            MemoryPool.GetFloatTensor(ref _yExpectedForEpoch, dataSet.Y_Shape);

            //we create the shuffled list of inputs 
            var shuffledElementId = Enumerable.Range(0, dataSet.Count).ToArray();
            if (epoch >= 2 && Config.RandomizeOrder && isTraining)
            {
                Utils.Shuffle(shuffledElementId, Config.Rand);
            }

            //we ensure that all slaves are on Idle state
            foreach (var slave in _slaveNetworks)
            {
                if (slave._slaveStatus != SLAVE_NETWORK_STATUS.IDLE)
                {
                    throw new Exception("must be in Idle status for network " + slave);
                }
            }

            StartTimer("WaitForSlave_Prepare", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
            SetStatusForAllSlaves(SLAVE_NETWORK_STATUS.PREPARE_MINIBATCH_GRADIENT_DESCENT);
            WaitForAllSlavesInStatus(SLAVE_NETWORK_STATUS.IDLE);
            StopTimer("WaitForSlave_Prepare", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);


            var x_miniBatch_cpu_allWorkers = new CpuTensor<float>(dataSet.XMiniBatch_Shape(miniBatchSizeForAllWorkers), null);
            var yExpected_miniBatch_cpu_allWorkers = new CpuTensor<float>(dataSet.YMiniBatch_Shape(miniBatchSizeForAllWorkers), null);


            for (int firstIndexInShuffledElementId_master= 0; firstIndexInShuffledElementId_master< totalElementCount; firstIndexInShuffledElementId_master+= miniBatchSizeForAllWorkers)
            {
                var currentMiniBatchSize_allWorkers = Math.Min(totalElementCount- firstIndexInShuffledElementId_master, miniBatchSizeForAllWorkers);
                var currentMiniBatchSize_master = Math.Min(miniBatchSizeForEachWorker, currentMiniBatchSize_allWorkers);
                Debug.Assert(currentMiniBatchSize_master>=1);
                x_miniBatch_cpu_allWorkers.Reshape(dataSet.XMiniBatch_Shape(currentMiniBatchSize_allWorkers));
                yExpected_miniBatch_cpu_allWorkers.Reshape(dataSet.YMiniBatch_Shape(currentMiniBatchSize_allWorkers));

                //we initialize miniBatch input (xMiniBatch) and expected output (yExpectedMiniBatchCpu)
                StartTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                bool withDataAugmentation = Config.DataAugmentation.UseDataAugmentation && (epoch >= 2) && isTraining;
                dataSet.LoadMiniBatch(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementId_master, Config.DataAugmentation, x_miniBatch_cpu_allWorkers, yExpected_miniBatch_cpu_allWorkers);
                StopTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                //we copy yExpected_miniBatch_cpu_allWorkers from CPU to appropriate target (CPU or GPU)
                var yExpectedForMiniBatch_allWorkers = _yExpectedForEpoch.RowSlice(firstIndexInShuffledElementId_master, currentMiniBatchSize_allWorkers);
                yExpected_miniBatch_cpu_allWorkers.CopyTo(yExpectedForMiniBatch_allWorkers);

                //we launch the forward & backward computation on all slave networks
                var usedSlaves = new List<Network>();
                int firstIndexInShuffledElement_slave = firstIndexInShuffledElementId_master + currentMiniBatchSize_master;
                foreach (var slave in _slaveNetworks)
                {
                    if (firstIndexInShuffledElement_slave >= totalElementCount)
                    {
                        break;
                    }
                    var currentMiniBatchSize_slave = Math.Min(totalElementCount - firstIndexInShuffledElement_slave, miniBatchSizeForEachWorker);
                    Debug.Assert(currentMiniBatchSize_slave >= 1);
                    var x_miniBatch_cpu_slave = x_miniBatch_cpu_allWorkers.RowSlice(firstIndexInShuffledElement_slave - firstIndexInShuffledElementId_master, currentMiniBatchSize_slave);
                    var yExpected_miniBatch_cpu_slave = yExpected_miniBatch_cpu_allWorkers.RowSlice(firstIndexInShuffledElement_slave - firstIndexInShuffledElementId_master, currentMiniBatchSize_slave);
                    var yPredicted_miniBatch_slave = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElement_slave, currentMiniBatchSize_slave);
                    slave._slaveParamForMiniBatchGradientDescent = Tuple.Create(x_miniBatch_cpu_slave, yExpected_miniBatch_cpu_slave, yPredicted_miniBatch_slave, isTraining);
                    slave._slaveStatus = SLAVE_NETWORK_STATUS.PERFORM_FORWARD_AND_BACKWARD_PROPAGATION;
                    firstIndexInShuffledElement_slave += currentMiniBatchSize_slave;
                    usedSlaves.Add(slave);
                }

                //we launch the forward & backward propagation on the master network
                var x_miniBatch_cpu_master = x_miniBatch_cpu_allWorkers.RowSlice(0, currentMiniBatchSize_master);
                MemoryPool.GetFloatTensor(ref _x_miniBatch, x_miniBatch_cpu_master.Shape);
                x_miniBatch_cpu_master.CopyTo(_x_miniBatch);
                var yPredicted_miniBatch_master = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElementId_master, currentMiniBatchSize_master);
                var yExpected_miniBatch_master = _yExpectedForEpoch.RowSlice(firstIndexInShuffledElementId_master, currentMiniBatchSize_master);
                PropagationManager.Forward(_x_miniBatch, yPredicted_miniBatch_master, isTraining);
                if (isTraining)
                {
                    PropagationManager.Backward(yExpected_miniBatch_master, yPredicted_miniBatch_master);
                }

                //we wait for all slave to finish the forward & backward propagation pass
                StartTimer("WaitForSlave_Forward", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                WaitForAllSlavesInStatus(SLAVE_NETWORK_STATUS.IDLE);
                StopTimer("WaitForSlave_Forward", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);

                if (isTraining)
                {
                    foreach (var usedSlave in usedSlaves)
                    {
                        StartTimer("CopyGradients", BackwardPropagationTime);
                        AddGradientFromSlaveNetwork(usedSlave);
                        StopTimer("CopyGradients", BackwardPropagationTime);
                    }
                    double percentagePerformedInEpoch = firstIndexInShuffledElementId_master / (double)totalElementCount;
                    PropagationManager.UpdateWeights(currentMiniBatchSize_allWorkers, learningRateComputerIfTraining.LearningRate(epoch, percentagePerformedInEpoch, lrMultiplicativeFactorFromReduceLrOnPlateau));
                }

                CallBackAfterEachMiniBatch?.Invoke(yExpected_miniBatch_master, yPredicted_miniBatch_master);

                if ((DateTime.Now-lastStatsUpdate).TotalSeconds> 5*60)
                {
                    var lastIndexInShuffledElementId = firstIndexInShuffledElementId_master + currentMiniBatchSize_allWorkers - 1;
                    var percentageDoneInEpoch = ((double) lastIndexInShuffledElementId) / totalElementCount;
                    var secondsSinceStartOfEpoch = (DateTime.Now - miniBatchGradientDescentStart).TotalSeconds;
                    var expectedSecondsToPerformEntireEpoch = secondsSinceStartOfEpoch / percentageDoneInEpoch;
                    Info("Epoch " + epoch + " in progress: " + Math.Round(100.0* percentageDoneInEpoch, 1) + "% performed ("+ Math.Round(secondsSinceStartOfEpoch, 0) + "s/"+Math.Round(expectedSecondsToPerformEntireEpoch,0)+"s)");
                    LogDebug(MemoryInfo());
                    lastStatsUpdate = DateTime.Now;
                }
            }

            x_miniBatch_cpu_allWorkers.Dispose();
            yExpected_miniBatch_cpu_allWorkers.Dispose();

            return _yPredictedForEpoch;
        }
        public int LastLayerIndex => Layers.Last().LayerIndex;

        private bool UseGPU => _resourceIds.Max() >= 0;
        private string MemoryInfo()
        {
            string result = "Private Memory: " + Utils.MemoryBytesToString((ulong)Process.GetCurrentProcess().PrivateMemorySize64);
            result += " - Managed Memory: " + Utils.MemoryBytesToString((ulong)GC.GetTotalMemory(false));
            result += " - " + MemoryPool;
            if (UseGPU)
            {
                result += " - " + GpuWrapper.MemoryInfo();
            }
            result += " - CurrentThreadId#" + Thread.CurrentThread.ManagedThreadId;
            return result;
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
            var x = mockMemoryPooling.GetFloatTensor(xShape);
            var yPredicted = mockMemoryPooling.GetFloatTensor(Layers.Last().OutputShape(x.Shape[0]));
            propagationManager.Forward(x, yPredicted, isTraining);
            if (isTraining)
            {
                var yExpected = yPredicted;
                propagationManager.Backward(yExpected, yPredicted);
            }
            return mockMemoryPooling.CapacityInBytes;
        }
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
            Debug.Assert(IsMaster);
            foreach (var l in Layers)
            {
                if (l != null && l.Trainable)
                {
                    l.ResetParameters();
                }
            }
        }
        private int MaxMiniBatchSizeForAllWorkers(int[] xShape, bool isTraining)
        {
            var bytesByBatchSizeForwardAndBackward = BytesByBatchSizeForwardAndBackward(xShape, isTraining);
            var freeMemoryInBytes = UseGPU ? (ulong)GpuWrapper.AvailableGpuMemoryInBytes() : Utils.AvailableRamMemoryInBytes();
            var maxMiniBatchSizeForEachWorker = MaxMiniBatchSizeForEachWorker(bytesByBatchSizeForwardAndBackward, freeMemoryInBytes);
            var maxMiniBatchSizeForAllWorkers = DegreeOfParallelism * maxMiniBatchSizeForEachWorker;
            LogDebug("Max MiniBatchSize=" + maxMiniBatchSizeForAllWorkers + " (free memory=" + Utils.MemoryBytesToString(freeMemoryInBytes) + ")");
            return maxMiniBatchSizeForAllWorkers;
        }
        //TODO add tests
        private static int MaxMiniBatchSizeForEachWorker(ulong bytesByBatchSize, ulong freeMemoryInBytes)
        {
            freeMemoryInBytes -= 1_500_000_000;
            var miniBatchSize = (int)(freeMemoryInBytes / (1.2*bytesByBatchSize));
            if (miniBatchSize > 4)
            {
                miniBatchSize -= (miniBatchSize % 4);
            }
            return miniBatchSize;
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
