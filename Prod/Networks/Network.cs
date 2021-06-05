using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Text;
using System.Threading;
using log4net;
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
        public static readonly ILog Log = LogManager.GetLogger(typeof(Network));
        public readonly List<EpochData> EpochData = new List<EpochData>();
        /// <summary>
        /// all resources (CPU or GPU) available for the current network
        /// values superior or equal to 0 means GPU resources (device)
        /// values strictly less then 0 mean CPU resources (host)
        /// </summary>
        private readonly List<int> _resourceIds;
        private Tensor buffer;
        private Tensor _bufferComputeLoss;
        private Tensor _randomNumberGeneratorStatesBufferForGPU;
        private readonly DateTime _timeStampCreation = DateTime.Now;
        // bytes/batchSize needed for forward & backward propagation
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
        /// <param name="resourceIds">list of resources available for the network
        /// if gpuDeviceIds.Count == 1
        ///     if masterNetworkIfAny == null:
        ///         all computation will be done in a single network (using resource gpuDeviceIds[0])
        ///     else:
        ///         we are in a slave network (using resource gpuDeviceIds[0]) doing part of the parallel computation
        ///         the master network is 'masterNetworkIfAny'.
        /// else: (gpuDeviceIds.Count >= 2)
        ///     we are the master network (using resource gpuDeviceIds[0]) doing part of the parallel computation
        ///     slaves network will use resourceId gpuDeviceIds[1:]
        ///
        /// for each resourceId in this list:
        ///     if resourceId strictly less then 0:
        ///         use CPU resource (no GPU usage)
        ///     else:
        ///         run the network on the GPU with device Id = resourceId
        /// </param>
        /// <param name="masterNetworkIfAny">
        ///     if the current network is a slave network doing computation for its master network:
        ///         the reference of the master network
        ///     else:
        ///         null
        /// </param>
        public Network(NetworkConfig config, List<int> resourceIds, Network masterNetworkIfAny = null)
        {
            //Utils.ConfigureGlobalLog4netProperties(config.LogDirectory, config.LogFile);
            Utils.ConfigureThreadLog4netProperties(config.LogDirectory, config.LogFile);

            //a slave network will have access to only one resource (1 Cpu or 1 GPU)
            Debug.Assert(masterNetworkIfAny == null || resourceIds.Count == 1);
            Config = config;

            _masterNetworkIfAny = masterNetworkIfAny;
            _resourceIds = resourceIds.ToList();
            GpuWrapper = UseGPU ? GPUWrapper.FromDeviceId(_resourceIds[0]) : null;
            _swComputeMetrics = new Stopwatch();
            CreateLogDirectoryIfNeeded();
            MemoryPool = new TensorMemoryPool(GpuWrapper);
            PropagationManager = new PropagationManager(Layers, MemoryPool, ForwardPropagationTrainingTime, ForwardPropagationInferenceTime, BackwardPropagationTime, _updateWeightsTime);
            if (IsMaster && _resourceIds.Count>=2)
            {
                //we create the slave networks
                foreach(var slaveResourceId in _resourceIds.Skip(1))
                {
                    Log.Debug("starting thread for network running on deviceId " + slaveResourceId);
                    new Thread(() => SlaveThread(this, slaveResourceId)).Start();
                }
                while (_slaveNetworks.Count != _resourceIds.Count - 1)
                {
                    Thread.Sleep(1);
                }
            }

        }

        public string DeviceName() { return GpuWrapper?.DeviceName(); }

        public void Dispose()
        {
            if (IsMaster)
            {
                Log.Debug("Before clearing memory: " + MemoryInfo());
                GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
                GC.Collect();
                SetStatusForAllSlaves(SLAVE_NETWORK_STATUS.TO_ABORT);
            }
            Layers.ForEach(l => l?.Dispose());
            Layers.Clear();
            PropagationManager.Dispose();
            EpochData.Clear();
            MemoryPool.FreeFloatTensor(ref buffer);
            MemoryPool.FreeFloatTensor(ref _bufferComputeLoss);
            MemoryPool.FreeFloatTensor(ref _randomNumberGeneratorStatesBufferForGPU);
            MemoryPool.FreeFloatTensor(ref _yPredictedForEpoch);
            MemoryPool.FreeFloatTensor(ref _yExpectedForEpoch);
            MemoryPool.FreeFloatTensor(ref _x_miniBatch);
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
                Log.Debug("After clearing memory: " + MemoryInfo());
            }
        }

        /// <summary>
        /// build a buffer needed by all Dropout layers when run on GPU
        /// this tensor is null for CPU
        /// </summary>
        /// <returns></returns>
        public Tensor GetRandomNumberGeneratorStatesBuffer()
        {
            if (!UseGPU)
            {
                return null;
            }
            if (_randomNumberGeneratorStatesBufferForGPU == null)
            {
                var res = CudnnWrapper.cudnnDropoutGetStatesSize(GpuWrapper.CudnnHandle, out var dropoutStateSize);
                GPUWrapper.CheckStatus(res);
                _randomNumberGeneratorStatesBufferForGPU = MemoryPool.GetBuffer(dropoutStateSize);
            }
            return _randomNumberGeneratorStatesBufferForGPU;
        }

        #region network construction: adding layers
        public Network Input(int[] shape_CHW, string layerName = "")
        {
            return Input(shape_CHW[0], shape_CHW[1], shape_CHW[2], layerName);
        }
        public Network Input(int channelCount, int h, int w, string layerName = "")
        {
            Debug.Assert(Layers.Count == 0);
            Layers.Add(new InputLayer(channelCount, h, w, this, layerName));
            return this;
        }
        public Network InputAndEmbedding(int maxWordsBySentence, int vocabularySize, int embeddingDim, int indexInLastDimensionToUse, double lambdaL2Regularization, string layerName = "")
        {
            Debug.Assert(Layers.Count == 0);
            Input(maxWordsBySentence, -1, -1);
            Embedding(vocabularySize, embeddingDim, indexInLastDimensionToUse, lambdaL2Regularization, layerName);
            return this;
        }
        public Network Embedding(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse, double lambdaL2Regularization, string layerName = "")
        {
            Debug.Assert(Layers.Count == 1);
            Layers.Add(new EmbeddingLayer(vocabularySize, embeddingDim, indexInLastDimensionToUse, lambdaL2Regularization, true, this, layerName));
            return this;
        }
        public Network SimpleRNN(int hiddenSize, bool returnSequences, bool isBidirectional, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            var simpleRnn = new RecurrentLayer(hiddenSize, cudnnRNNMode_t.CUDNN_RNN_TANH, cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS, returnSequences, isBidirectional, true, this, layerName);
            Layers.Add(simpleRnn);
            return this;
        }
        public Network LSTM(int units, bool returnSequences, bool isBidirectional, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Debug.Assert(UseGPU);
            var lstm = new RecurrentLayer(units, cudnnRNNMode_t.CUDNN_LSTM, cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS, returnSequences, isBidirectional, true, this, layerName);
            Layers.Add(lstm);
            return this;
        }

        public Network GRU(int hiddenSize, bool returnSequences, bool isBidirectional, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Debug.Assert(UseGPU);
            var lstm = new RecurrentLayer(hiddenSize, cudnnRNNMode_t.CUDNN_GRU, cudnnRNNBiasMode_t.CUDNN_RNN_DOUBLE_BIAS, returnSequences, isBidirectional, true, this, layerName);
            Layers.Add(lstm);
            return this;
        }

        public Network Dense(int units, double lambdaL2Regularization, bool flattenInputTensorOnLastDimension, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            var fullyConnectedLayer = new DenseLayer(units, lambdaL2Regularization, flattenInputTensorOnLastDimension, true, this, layerName);
            Layers.Add(fullyConnectedLayer);
            return this;
        }
        // ReSharper disable once UnusedMethodReturnValue.Global
        public Network Linear(float slope, float intercept, string layerName = "")
        {
            var linearFunctionLayer = new LinearFunctionLayer(slope, intercept, this, layerName);
            Layers.Add(linearFunctionLayer);
            return this;
        }
        // ReSharper disable once UnusedMethodReturnValue.Global
        public Network CustomLinear(float slopeConstant, string layerName = "")
        {
            var linearFunctionLayer = new CustomLinearFunctionLayer(slopeConstant, this, layerName);
            Layers.Add(linearFunctionLayer);
            return this;
        }

        public Network Convolution_BatchNorm(int filtersCount, int kernelSize, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization)
        {
            return Convolution(filtersCount, kernelSize, stride, paddingType, lambdaL2Regularization, false)
                .BatchNorm(0.99, 1e-5);
        }
        public Network Convolution_BatchNorm_Activation(int filtersCount, int kernelSize, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, cudnnActivationMode_t activationFunction)
        {
            return Convolution_BatchNorm(filtersCount, kernelSize, stride, paddingType, lambdaL2Regularization)
                .Activation(activationFunction);
        }
        // ReSharper disable once UnusedMethodReturnValue.Global
        public Network BatchNorm_Activation(cudnnActivationMode_t activationFunction)
        {
            return BatchNorm(0.99, 1e-5).Activation(activationFunction);
        }
        public Network BatchNorm_Activation_Convolution(cudnnActivationMode_t activationFunction, int filtersCount, int kernelSize, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias)
        {
            return 
                BatchNorm(0.99, 1e-5)
                .Activation(activationFunction)
                .Convolution(filtersCount, kernelSize, stride, paddingType, lambdaL2Regularization, useBias);
        }
        public Network AddLayer(int previousResidualLayerIndex, int previousIdentityLayerIndex, string layerName = "")
        {
            Layers.Add(new AddLayer(new []{previousResidualLayerIndex, previousIdentityLayerIndex}, this, layerName));
            Debug.Assert(Layers[previousIdentityLayerIndex].SameOutputShape(Layers[previousResidualLayerIndex]));
            return this;
        }
        public Network ConcatenateLayer(int previousLayerIndex1, int previousLayerIndex2, string layerName = "")
        {
            return ConcatenateLayer(new []{previousLayerIndex1, previousLayerIndex2}, layerName);
        }
        public Network ConcatenateLayer(int[] previousLayers, string layerName = "")
        {
            Layers.Add(new ConcatenateLayer(previousLayers, this, layerName));
            return this;
        }
        public Network MultiplyLayer(int previousLayerIndex1, int previousLayerIndexDiagonalMatrix, string layerName = "")
        {
            Layers.Add(new MultiplyLayer(previousLayerIndex1, previousLayerIndexDiagonalMatrix, this, layerName));
            return this;
        }
        // ReSharper disable once UnusedMethodReturnValue.Global
        public Network NonMaxSuppression(float minScore, float IOU_threshold_for_duplicate, int maxOutputSize, int maxOutputSizePerClass, string layerName)
        {
            Layers.Add(new NonMaxSuppressionLayer(maxOutputSizePerClass, maxOutputSize, IOU_threshold_for_duplicate, minScore, this, layerName));
            return this;
        }
        public Network UpSampling2D(int rowFactor, int colFactor, UpSampling2DLayer.InterpolationEnum interpolation, string layerName = "")
        {
            Layers.Add(new UpSampling2DLayer(rowFactor, colFactor, interpolation, this, layerName));
            return this;
        }
        // ReSharper disable once UnusedMethodReturnValue.Global
        public Network YOLOV3Layer(int[] anchors, int previousLayerIndex, string layerName)
        {
            Layers.Add(new YOLOV3Layer(anchors, previousLayerIndex, this, layerName));
            return this;
        }
        public Network ZeroPadding2D(int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, string layerName = "")
        {
            return ZeroPadding2D(paddingTop, paddingBottom, paddingLeft, paddingRight, Layers.Count-1, layerName);
        }
        public Network ZeroPadding2D(int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int previousLayerIndex, string layerName = "")
        {
            Layers.Add(new ZeroPadding2DLayer(paddingTop, paddingBottom, paddingLeft, paddingRight, previousLayerIndex, this, layerName));
            return this;
        }

        //add a shortcut from layer 'AddSumLayer' to current layer, adding a Conv Layer if necessary (for matching size)
        // ReSharper disable once UnusedMethodReturnValue.Global
        public Network Shortcut_IdentityConnection(int startOfBlockLayerIndex, int filtersCount, int stride, double lambdaL2Regularization)
        {
            int previousResidualLayerIndex = LastLayerIndex;

            var sameInputAndOutputShapeInBlock = Layers.Last().SameOutputShape(Layers[startOfBlockLayerIndex]);
            if (sameInputAndOutputShapeInBlock)
            {
                Layers.Add(new AddLayer(new []{previousResidualLayerIndex, startOfBlockLayerIndex}, this));
            }
            else
            {
                //we need to add a convolution layer to make correct output format
                Convolution(filtersCount, 1, stride, 0, lambdaL2Regularization, true, startOfBlockLayerIndex);
                int convLayerIdInIdentityBlock = LastLayerIndex;
                Layers.Add(new AddLayer(new[]{previousResidualLayerIndex, convLayerIdInIdentityBlock}, this));
                Debug.Assert(Layers[convLayerIdInIdentityBlock].SameOutputShape(Layers[previousResidualLayerIndex]));
            }
            return this;
        }
        public Network Convolution(int filtersCount, int kernelSize, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, string layerName = "")
        {
            return Convolution(filtersCount, kernelSize, stride, paddingType, lambdaL2Regularization, useBias, Layers.Count - 1, layerName);
        }
        public Network Convolution(int filtersCount, int kernelSize, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, int previousLayerIndex, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ConvolutionLayer(false, false, filtersCount, -1, kernelSize, kernelSize, stride, paddingType, lambdaL2Regularization, useBias, previousLayerIndex, true, this, layerName));
            return this;
        }

        public Network Conv1D(int filtersCount, int kernelWidth, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, string layerName = "")
        {
            return Conv1D(filtersCount, kernelWidth, stride, paddingType, lambdaL2Regularization, useBias, Layers.Count - 1, layerName);
        }
        private Network Conv1D(int filtersCount, int kernelWidth, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, int previousLayerIndex, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ConvolutionLayer(false, true, filtersCount, -1, 1, kernelWidth, stride, paddingType, lambdaL2Regularization, useBias, previousLayerIndex, true, this, layerName));
            return this;
        }
        public Network DepthwiseConvolution(int kernelSize, int stride, ConvolutionLayer.PADDING_TYPE paddingType, int depthMultiplier, double lambdaL2Regularization, bool useBias, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ConvolutionLayer(true, false, -1, depthMultiplier, kernelSize, kernelSize, stride, paddingType, lambdaL2Regularization, useBias, Layers.Count - 1, true, this, layerName));
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
            return Activation(activationFunction, null, layerName);
        }
        public Network Activation(cudnnActivationMode_t activationFunction, Tensor activationParameter, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ActivationLayer(activationFunction, activationParameter, this, layerName));
            return this;
        }

        public Network MaxPooling(int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, string layerName = "")
        {
            return MaxPooling(poolingHeight, poolingWidth, verticalStride, horizontalStride, Layers.Count-1, layerName);
        }
        private Network MaxPooling(int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int previousLayerIndex, string layerName)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new PoolingLayer(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingHeight, poolingWidth, verticalStride, horizontalStride, previousLayerIndex, this, layerName));
            return this;
        }
        public Network AvgPooling(int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            int previousLayerIndex = Layers.Count-1;
            Layers.Add(new PoolingLayer(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, poolingHeight, poolingWidth, verticalStride, horizontalStride, previousLayerIndex, this, layerName));
            return this;
        }
        public Network GlobalAvgPooling(string layerName = "")
        {
            return AvgPooling(-1, -1, -1, -1, layerName);
        }

        /// <summary>
        /// Global Average Pooling on the 2nd from last dimension of a tensor
        /// It will transform a tensor from shape (n, ..., height, width) to shape (n, ..., 1, width)
        /// </summary>
        public Network GlobalAvgPoolingOnHeight(string layerName = "")
        {
            return AvgPooling(-1, 1, -1, 1, layerName);
        }
        /// <summary>
        /// Global Average Pooling on the last dimension of a tensor
        /// It will transform a tensor from shape (n, ..., height, width) to shape (n, ..., height, 1)
        /// </summary>
        // ReSharper disable once UnusedMember.Global
        public Network GlobalAvgPoolingOnWidth(string layerName = "")
        {
            return AvgPooling(1, -1, 1, -1, layerName);
        }
        // ReSharper disable once UnusedMethodReturnValue.Global
        public Network GlobalMaxPooling(string layerName = "")
        {
            return GlobalMaxPooling(Layers.Count - 1, layerName);
        }
        public Network GlobalMaxPooling(int previousLayerIndex, string layerName = "")
        {
            return MaxPooling(-1, -1, -1, previousLayerIndex, layerName);
        }
        // ReSharper disable once UnusedMethodReturnValue.Global
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
            Layers.Add(new BatchNormalizationLayer(momentum, epsilon, true, this, layerName));
            return this;
        }
        public Network Dense_Activation(int units, double lambdaL2Regularization, bool flattenInputTensorOnLastDimension, cudnnActivationMode_t activationFunction)
        {
            return Dense(units, lambdaL2Regularization, flattenInputTensorOnLastDimension)
                .Activation(activationFunction);
        }
        public Network Dense_DropOut_Activation(int units, double lambdaL2Regularization, double dropOut, cudnnActivationMode_t activationFunction)
        {
            return Dense(units, lambdaL2Regularization, false)
                .Dropout(dropOut)
                .Activation(activationFunction);
        }
        public Network Output(int units, double lambdaL2Regularization, cudnnActivationMode_t activationFunctionType)
        {
            return Dense(units, lambdaL2Regularization, false)
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
            return Summary();
        }
      
        public int TotalParams => Layers.SelectMany(l => l.Parameters).Select(t=> t.Item1.Count).Sum();
        private int NonTrainableParams => Layers.Select(l => l.NonTrainableParams).Sum();
        // ReSharper disable once UnusedMethodReturnValue.Global
        public double FindBestLearningRate(IDataSet trainingDataSet, double minLearningRate, double maxLearningRate, int miniBatchSizeForAllWorkers)
        {
            Debug.Assert(minLearningRate >= 0);
            Debug.Assert(maxLearningRate >= 0);
            Debug.Assert(maxLearningRate > minLearningRate);
            Debug.Assert(miniBatchSizeForAllWorkers >= 1);

            Log.Info(ToString());
            Log.Info("Looking for best learning rate...");
            ResetWeights(); //restore weights to their original values
            Log.Debug("BatchSize: "+ miniBatchSizeForAllWorkers);
            var learningRateFinder = new LearningRateFinder(miniBatchSizeForAllWorkers, trainingDataSet.Count, minLearningRate, maxLearningRate);

            void CallBackAfterEachMiniBatch(Tensor yExpectedMiniBatch, Tensor yPredictedMiniBatch)
            {
                MemoryPool.GetFloatTensor(ref _bufferComputeLoss, new[] { yExpectedMiniBatch.Shape[0] });
                var blockLoss = yExpectedMiniBatch.ComputeLoss(yPredictedMiniBatch, Config.LossFunction, _bufferComputeLoss);
                learningRateFinder.AddLossForLastBlockId(blockLoss);
            }
            MiniBatchGradientDescentForSingleEpoch(trainingDataSet, miniBatchSizeForAllWorkers, learningRateFinder, CallBackAfterEachMiniBatch);
            var fileName = Path.Combine(Config.LogDirectory, UniqueId + "_LearningRateFinder.csv");
            File.WriteAllText(fileName, learningRateFinder.AsCsv());
            Log.Info("Stats stored in: " + fileName);
            var bestLearningRate = learningRateFinder.BestLearningRate();
            Log.Info("Best learning rate: "+ bestLearningRate+ " (with batch size="+miniBatchSizeForAllWorkers+")");
            ResetWeights(); //restore weights to there original values
            return bestLearningRate;
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
        /// <param name="miniBatchSizeForAllWorkers"></param>
        /// <param name="testDataSetCpuIfAny"></param>
        public void Fit(IDataSet trainingDataSetCpu, ILearningRateComputer learningRateComputer, int numEpochs, int miniBatchSizeForAllWorkers, IDataSet testDataSetCpuIfAny)
        {
            try
            {
                Debug.Assert(Config.TypeSize == trainingDataSetCpu.TypeSize);
                Debug.Assert(miniBatchSizeForAllWorkers >= 1);
                if (learningRateComputer == null)
                {
                    throw new ArgumentException("learningRateComputer shouldn't be null in Training mode");
                }
                _spInternalFit.Start();
                StartTimer("Fit_Prepare", ForwardPropagationTrainingTime);

                FreezeSelectedLayers();

                Log.Debug("Fit( " + Tensor.ShapeToString(XMiniBatch_Shape(trainingDataSetCpu.Count))+" => " + Tensor.ShapeToString(trainingDataSetCpu.YMiniBatch_Shape(trainingDataSetCpu.Count))+" )");
                Log.Info(ToString());


                if (UseGPU)
                {
                    Log.Debug(GpuWrapper.ToString());
                }
                Log.Debug("Training Set: " + trainingDataSetCpu);
                if (testDataSetCpuIfAny != null)
                {
                    Log.Debug("Test Set: " + testDataSetCpuIfAny);
                }
                Log.Info("#Epochs=" + numEpochs + " BatchSize=" + miniBatchSizeForAllWorkers+" Name="+Description);
                if (Config.DisplayTensorContentStats)
                {
                    Log.Debug("Initial Tensor Content stats" + Environment.NewLine + ContentStats() + Environment.NewLine);
                }

                //Info(GpuWrapper.ToString());

                StopTimer("Fit_Prepare", ForwardPropagationTrainingTime);


                var lastAutoSaveTime = DateTime.Now; //last time we saved the network
                IDictionary<NetworkConfig.Metric, double> validationMetrics = null;
                for (;;)
                {
                    int epoch = EpochData.Count + 1;
                    if (epoch > numEpochs)
                    {
                        break;
                    }

                    var swEpoch = Stopwatch.StartNew();

                    var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputer.MultiplicativeFactorFromReduceLrOnPlateau(EpochData);
                    if (learningRateComputer.ShouldReduceLrOnPlateau(EpochData))
                    {
                        Log.Info("Reducing learningRate because of plateau at epoch " + epoch + " (new multiplicative coeff:"+ lrMultiplicativeFactorFromReduceLrOnPlateau+")");
                    }

                    #region Mini Batch gradient descent
                    var learningRateAtEpochStart = learningRateComputer.LearningRate(epoch, 0, lrMultiplicativeFactorFromReduceLrOnPlateau);
                    var yPredicted = MiniBatchGradientDescentForSingleEpoch(trainingDataSetCpu, miniBatchSizeForAllWorkers, learningRateComputer, null);
                    #endregion

                    //We display stats about the just finished epoch
                    if (Config.DisplayTensorContentStats)
                    {
                        Log.Debug("End of Epoch:" + epoch + " Tensor Content stats" + Environment.NewLine+ContentStats()+Environment.NewLine);
                    }

                    StartTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);
                    var trainingMetricsForEpoch = ComputeMetrics(_yExpectedForEpoch, yPredicted);
                    var lossAndAccuracyMsg = MetricsToString(trainingMetricsForEpoch, "");
                    if (testDataSetCpuIfAny != null)
                    {
                        //We compute the validation (= test) loss&accuracy
                        if (ShouldUseFullTestDataSetForLossAndAccuracy(learningRateComputer, epoch, numEpochs))
                        {
                            validationMetrics = ComputeMetricsForTestDataSet(miniBatchSizeForAllWorkers, testDataSetCpuIfAny);
                            lossAndAccuracyMsg += " - " + MetricsToString(validationMetrics, "val_");
                        }
                        else
                        {
                            //we'll compute loss and accuracy using only 10% of the test data set
                            using var subDataSet = testDataSetCpuIfAny.SubDataSet(i => i%10 == 0);
                            validationMetrics = ComputeMetricsForTestDataSet(miniBatchSizeForAllWorkers, subDataSet);
                            lossAndAccuracyMsg += " - " + MetricsToString(validationMetrics, "estimate_val_");
                        }

                    }
                    StopTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);

                    double secondsForEpoch = swEpoch.Elapsed.TotalSeconds;
                    double nbStepsByEpoch = ((double)trainingDataSetCpu.Count) / miniBatchSizeForAllWorkers;
                    var msByStep = (1000 * secondsForEpoch) / nbStepsByEpoch;
                    Log.Info("Epoch " + epoch + "/" + numEpochs + " - " + Math.Round(secondsForEpoch, 0) + "s " + Math.Round(msByStep, 0) + "ms/step - lr: "+Math.Round(learningRateAtEpochStart, 8)+" - "+lossAndAccuracyMsg+" - "+ Description);
                    Log.Debug(MemoryInfo());
                    //if it is the last epoch, we'll save Layer KPI
                    if (epoch == numEpochs)
                    {
                        Log.Debug(LayersKpi());
                    }

                    #region we save stats about the just finished epoch
                    var currentEpochData = new EpochData(epoch, learningRateAtEpochStart, lrMultiplicativeFactorFromReduceLrOnPlateau, secondsForEpoch, trainingMetricsForEpoch, validationMetrics);
                    EpochData.Add(currentEpochData);
                    #endregion

                    #region we save the network in a file if necessary
                    if (   //if we have finished training
                           ((epoch == numEpochs) && (numEpochs > 10))
                            //or if we should save the network every 'Config.AutoSaveIntervalInMinutes' minutes
                        || ( (Config.AutoSaveIntervalInMinutes>=0) && (DateTime.Now - lastAutoSaveTime).TotalMinutes > Config.AutoSaveIntervalInMinutes)
                        || learningRateComputer.ShouldCreateSnapshotForEpoch(epoch)
                        )
                    {
                        var modelFilePath = Path.Combine(Config.LogDirectory, UniqueId + "_" + epoch + ".txt");
                        var parametersFilePath = ModelFilePath2ParameterFilePath(modelFilePath);
                        SaveModelAndParameters(modelFilePath, parametersFilePath);
                        lastAutoSaveTime = DateTime.Now;
                    }
                    #endregion

                    if (Config.SaveNetworkStatsAfterEachEpoch)
                    {
                        var networkStatFileName = Path.Combine(Config.LogDirectory, UniqueId + "_" + epoch + "_NetworkStats.txt");
                        Log.Info("Saving network '" + Description + "' stats in " + networkStatFileName);
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
                        + EpochData.Last().TrainingLoss + ";"
                        + EpochData.Last().TrainingAccuracy + ";"
                        + EpochData.Last().ValidationLoss + ";"
                        + EpochData.Last().ValidationAccuracy + ";"
                        + Environment.NewLine;
                    var testsCsv = string.IsNullOrEmpty(trainingDataSetCpu.Name)?"Tests.csv": ("Tests_"+ trainingDataSetCpu.Name + ".csv");
                    if (Config.LogEnabled)
                    {
                        File.AppendAllText(Utils.ConcatenatePathWithFileName(Config.LogDirectory, testsCsv), line);
                    }
                }
                catch (Exception e)
                {
                    Log.Info("fail to add line in file:" + Environment.NewLine + line + Environment.NewLine + e);
                    // ignored
                }

                Log.Info("Training '"+ Description+"' for " + numEpochs + " epochs took: " + _spInternalFit.Elapsed.TotalSeconds + "s");
                if (!string.IsNullOrEmpty(Description))
                {
                    Log.Debug("Network Name: "+Description);
                }
                _spInternalFit.Stop();
            }
            catch (Exception e)
            {
                Log.Error(e.ToString());
                throw;
            }
        }
        
        private bool ShouldUseFullTestDataSetForLossAndAccuracy(ILearningRateComputer learningRateComputer, int epoch, int numEpoch)
        {
            if (Config.AlwaysUseFullTestDataSetForLossAndAccuracy || epoch == 1 || epoch == numEpoch)
            {
                return true;
            }
            return learningRateComputer.ShouldCreateSnapshotForEpoch(epoch);
        }

        #region compute Loss and Accuracy
        public IDictionary<NetworkConfig.Metric, double> ComputeMetricsForTestDataSet(int miniBatchSize, IDataSet testDataSet)
        {
            //We perform a mini batch gradient descent in Testing mode:
            //  there will be no shuffling/data augmentation.
            var yPredicted = MiniBatchGradientDescentForSingleEpoch(testDataSet, miniBatchSize);
            return ComputeMetrics(testDataSet.Y, yPredicted);
        }

        private IDictionary<NetworkConfig.Metric, double> ComputeMetrics(Tensor yExpected, Tensor yPredicted)
        {
            _swComputeMetrics?.Start();
            var result = new Dictionary<NetworkConfig.Metric, double>();
            yExpected = ReformatToCorrectDevice_GPU_or_CPU(yExpected);
            yPredicted = ReformatToCorrectDevice_GPU_or_CPU(yPredicted);
            var batchSize = yExpected.Shape[0];
            foreach (var metric in Config.Metrics)
            {
                if (metric == NetworkConfig.Metric.Loss)
                {
                    MemoryPool.GetFloatTensor(ref _bufferComputeLoss, new[] {batchSize});
                    result[metric] = yExpected.ComputeLoss(yPredicted, Config.LossFunction, _bufferComputeLoss);
                }
                else if (metric == NetworkConfig.Metric.Accuracy)
                {
                    MemoryPool.GetFloatTensor(ref buffer, new[] { batchSize });
                    result[metric] = yExpected.ComputeAccuracy(yPredicted, Config.LossFunction, buffer);
                }
                else if (metric == NetworkConfig.Metric.Mae)
                {
                    MemoryPool.GetFloatTensor(ref buffer, new[] { batchSize });
                    result[metric] = yExpected.ComputeMae(yPredicted, buffer);
                }
                else if (metric == NetworkConfig.Metric.Mse)
                {
                    MemoryPool.GetFloatTensor(ref buffer, new[] { batchSize });
                    result[metric] = yExpected.ComputeMse(yPredicted, buffer);
                }
                else
                {
                    throw new ArgumentException("unknown metric "+metric);
                }
            }
            _swComputeMetrics?.Stop();
            return result;
        }
        public static string MetricsToString(IDictionary<NetworkConfig.Metric, double> metrics, string prefix)
        {
            return string.Join(" - ", metrics.OrderBy(x => x.Key).Select(e => prefix + e.Key + ": " + Math.Round(e.Value, 4))).ToLowerInvariant();
        }
        public static string TrainingAndValidationMetricsToString(IDictionary<NetworkConfig.Metric, double> trainingMetrics, IDictionary<NetworkConfig.Metric, double> validationMetrics)
        {
            return MetricsToString(trainingMetrics, "") + " - " + MetricsToString(validationMetrics, "val_");
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
        }

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
            var yPredicted = MemoryPool.GetFloatTensor(Layers.Last().OutputShape(X.Shape[0]));
            X = ReformatToCorrectDevice_GPU_or_CPU(X);
            PropagationManager.Forward(X, yPredicted, isTraining, null, null);
            return yPredicted;
        }

        public CpuTensor<float> Predict(IDataSet dataSet, int miniBatchSizeForAllWorkers)
        {
            var yPredicted = MiniBatchGradientDescentForSingleEpoch(dataSet, miniBatchSizeForAllWorkers);
            var yPredictedCpu = yPredicted.ToCpuFloat();
            return yPredictedCpu;
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
        public Tensor MiniBatchGradientDescentForSingleEpoch(IDataSet dataSet, int miniBatchSizeForAllWorkers, ILearningRateComputer learningRateComputerIfTraining = null, Action<Tensor, Tensor> CallBackAfterEachMiniBatch = null)
        {
            Debug.Assert(IsMaster);
            Debug.Assert(miniBatchSizeForAllWorkers >= 1);

            if (_slaveNetworks.Any())
            {
                CompactParameters();
                CompactGradients();
            }

            //last time we display a progress on the screen for the current min batch descent
            var miniBatchGradientDescentStart = DateTime.Now;
            var lastStatsUpdate = miniBatchGradientDescentStart;
            bool isTraining = learningRateComputerIfTraining != null;

            //the mini batch size must be a multiple of the number of workers
            Debug.Assert(miniBatchSizeForAllWorkers< DegreeOfParallelism || miniBatchSizeForAllWorkers % DegreeOfParallelism == 0);
            int miniBatchSizeForEachWorker = Math.Max(1, miniBatchSizeForAllWorkers / DegreeOfParallelism);

            //the first epoch is #1
            int epoch = EpochData.Count + 1;
            var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputerIfTraining?.MultiplicativeFactorFromReduceLrOnPlateau(EpochData) ?? 1.0;

            //dataSet.Count:
            // actual number of elements in the dataSet that we'll process
            //dataSetCountWithExtraBufferAtEnd:
            // Length of the tensors used to store the expected (_yExpectedForEpoch) & predicted values (_yPredictedForEpoch)
            // Those tensors contains an extra buffer of 'miniBatchSizeForAllWorkers-1' elements at the end
            // to make sure we can always split the dataSet in batch of exactly 'miniBatchSizeForAllWorkers' elements
            int dataSetCountWithExtraBufferAtEnd = dataSet.Count + miniBatchSizeForAllWorkers - 1;
            var YShapeAsMultipleOfMiniBatchSize = dataSet.YMiniBatch_Shape(dataSetCountWithExtraBufferAtEnd);
            MemoryPool.GetFloatTensor(ref _yPredictedForEpoch, YShapeAsMultipleOfMiniBatchSize);
            MemoryPool.GetFloatTensor(ref _yExpectedForEpoch, YShapeAsMultipleOfMiniBatchSize);

            //we create the shuffled list of inputs 
            var shuffledElementId = Enumerable.Range(0, dataSetCountWithExtraBufferAtEnd).ToArray();
            for (int i = dataSet.Count; i < shuffledElementId.Length; ++i)
            {
                shuffledElementId[i] = i%dataSet.Count;
            }

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

            var x_miniBatch_cpu_allWorkers = new CpuTensor<float>(XMiniBatch_Shape(miniBatchSizeForAllWorkers), null);
            var yExpected_miniBatch_cpu_allWorkers = new CpuTensor<float>(dataSet.YMiniBatch_Shape(miniBatchSizeForAllWorkers), null);

            var shuffledElementIdMemory = new Memory<int>(shuffledElementId);
            for (int firstIndexInShuffledElementId = 0; firstIndexInShuffledElementId < dataSet.Count; )
            {
                //Log.Debug("Processing epoch " + epoch + " for elements [" + firstIndexInShuffledElementId + "," + (firstIndexInShuffledElementId_master + currentMiniBatchSize_master - 1) + "]");

                //we initialize miniBatch input (xMiniBatch) and expected output (yExpectedMiniBatchCpu)
                StartTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                bool withDataAugmentation = Config.DataAugmentation.UseDataAugmentation && (epoch >= 2) && isTraining;
                int actualNumberOfLoadedItems = dataSet.LoadMiniBatch(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementId, Config.DataAugmentation, x_miniBatch_cpu_allWorkers, yExpected_miniBatch_cpu_allWorkers);
                StopTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                //we copy yExpected_miniBatch_cpu_allWorkers from CPU to appropriate target (CPU or GPU)
                var yExpectedForMiniBatch_allWorkers = _yExpectedForEpoch.RowSlice(firstIndexInShuffledElementId, miniBatchSizeForAllWorkers);
                yExpected_miniBatch_cpu_allWorkers.CopyTo(yExpectedForMiniBatch_allWorkers);

                //we launch the forward & backward computation on all slave networks
                var usedSlaves = new List<Network>();
                int firstIndexInShuffledElement_slave = firstIndexInShuffledElementId + miniBatchSizeForEachWorker;
                foreach (var slave in _slaveNetworks)
                {
                    var x_miniBatch_cpu_slave = x_miniBatch_cpu_allWorkers.RowSlice(firstIndexInShuffledElement_slave - firstIndexInShuffledElementId, miniBatchSizeForEachWorker);
                    var yExpected_miniBatch_cpu_slave = yExpected_miniBatch_cpu_allWorkers.RowSlice(firstIndexInShuffledElement_slave - firstIndexInShuffledElementId, miniBatchSizeForEachWorker);
                    var yPredicted_miniBatch_slave = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElement_slave, miniBatchSizeForEachWorker);
                    var elementIdsInDataSet_slave = shuffledElementIdMemory.Slice(firstIndexInShuffledElement_slave, miniBatchSizeForEachWorker);
                    slave._slaveParamForMiniBatchGradientDescent = Tuple.Create(x_miniBatch_cpu_slave, yExpected_miniBatch_cpu_slave, yPredicted_miniBatch_slave, isTraining, dataSet, elementIdsInDataSet_slave);
                    slave._slaveStatus = SLAVE_NETWORK_STATUS.PERFORM_FORWARD_AND_BACKWARD_PROPAGATION;
                    firstIndexInShuffledElement_slave += miniBatchSizeForEachWorker;
                    usedSlaves.Add(slave);
                }

                //we launch the forward & backward propagation on the master network
                var x_miniBatch_cpu_master = x_miniBatch_cpu_allWorkers.RowSlice(0, miniBatchSizeForEachWorker);
                MemoryPool.GetFloatTensor(ref _x_miniBatch, x_miniBatch_cpu_master.Shape);
                x_miniBatch_cpu_master.CopyTo(_x_miniBatch);
                var yPredicted_miniBatch_master = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElementId, miniBatchSizeForEachWorker);
                var yExpected_miniBatch_master = _yExpectedForEpoch.RowSlice(firstIndexInShuffledElementId, miniBatchSizeForEachWorker);
                var elementIdsInDataSet_master = shuffledElementIdMemory.Slice(firstIndexInShuffledElementId, miniBatchSizeForEachWorker);
                PropagationManager.Forward(_x_miniBatch, yPredicted_miniBatch_master, isTraining, dataSet, elementIdsInDataSet_master);
                if (isTraining)
                {
                    PropagationManager.Backward(yExpected_miniBatch_master, yPredicted_miniBatch_master, Config.LossFunction);
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
                    double percentagePerformedInEpoch = firstIndexInShuffledElementId / (double) dataSetCountWithExtraBufferAtEnd;
                    PropagationManager.UpdateWeights(miniBatchSizeForAllWorkers, learningRateComputerIfTraining.LearningRate(epoch, percentagePerformedInEpoch, lrMultiplicativeFactorFromReduceLrOnPlateau));

                }

                if (!isTraining && dataSet is ITimeSeriesDataSet)
                {
                    //During inference for TimeSeries, we'll need the previous predicted values to predict the next one
                    var batchPredictions = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElementId, actualNumberOfLoadedItems);
                    var batchElementIds = shuffledElementIdMemory.Slice(firstIndexInShuffledElementId, actualNumberOfLoadedItems).ToArray();
                    ((ITimeSeriesDataSet)dataSet).SetBatchPredictionsForInference(batchElementIds, batchPredictions);
                }

                CallBackAfterEachMiniBatch?.Invoke(yExpected_miniBatch_master, yPredicted_miniBatch_master);

                if ((DateTime.Now-lastStatsUpdate).TotalSeconds> 10*60)
                {
                    var lastIndexInShuffledElementId = firstIndexInShuffledElementId + miniBatchSizeForAllWorkers - 1;
                    var percentageDoneInEpoch = ((double) lastIndexInShuffledElementId) / dataSetCountWithExtraBufferAtEnd;
                    var secondsSinceStartOfEpoch = (DateTime.Now - miniBatchGradientDescentStart).TotalSeconds;
                    var expectedSecondsToPerformEntireEpoch = secondsSinceStartOfEpoch / percentageDoneInEpoch;
                    Log.Info((isTraining ? ("Epoch " + epoch) : "Inference") + " in progress: " + Math.Round(100.0 * percentageDoneInEpoch, 1) + "% performed (" + Math.Round(secondsSinceStartOfEpoch, 0) + "s/" + Math.Round(expectedSecondsToPerformEntireEpoch, 0) + "s)");
                    Log.Debug(MemoryInfo());
                    lastStatsUpdate = DateTime.Now;
                }
                firstIndexInShuffledElementId += actualNumberOfLoadedItems;
            }

            x_miniBatch_cpu_allWorkers.Dispose();
            yExpected_miniBatch_cpu_allWorkers.Dispose();

            _yPredictedForEpoch.Reshape(dataSet.Y.Shape);
            _yExpectedForEpoch.Reshape(dataSet.Y.Shape);

            return _yPredictedForEpoch;
        }
        public int LastLayerIndex => Layers.Last().LayerIndex;

        public int NbLayerOfType(Type layerType)
        {
            return Layers.Count(l => l.GetType() == layerType);
        }


        public bool UseGPU => _resourceIds.Max() >= 0;
        private string MemoryInfo()
        {
            string result = "Private Memory: " + Utils.MemoryBytesToString((ulong)Process.GetCurrentProcess().PrivateMemorySize64);
            result += " - Managed Memory: " + Utils.MemoryBytesToString((ulong)GC.GetTotalMemory(false));
            result += " - " + MemoryPool;
            if (UseGPU)
            {
                result += " - " + GpuWrapper.MemoryInfo();
            }
            return result;
        }


        private int[] XMiniBatch_Shape(int miniBatchSize)
        {
            Debug.Assert(Layers.Count>=1);
            Debug.Assert(Layers[0] is InputLayer);
            return ((InputLayer) Layers[0]).OutputShape(miniBatchSize);
        }


        public string UniqueId => (string.IsNullOrEmpty(Description) ? "Network" : Utils.ToValidFileName(Description)) + "_" + _timeStampCreation.ToString("yyyyMMdd_HHmm", CultureInfo.InvariantCulture);
        private void CreateLogDirectoryIfNeeded()
        {
            if (!string.IsNullOrEmpty(Config.LogDirectory) && !Directory.Exists(Config.LogDirectory))
            {
                Directory.CreateDirectory(Config.LogDirectory);
            }
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
                var firstColumn = layer.LayerName + " (" + layer.LayerType() + ")";
                if (firstColumn.Length > firstColumnWidth - 1)
                {
                    firstColumn = firstColumn.Substring(0, firstColumnWidth - 1);
                }
                var previousLayers = layer.PreviousLayers.ToList();
                var firstPreviousLayer = (previousLayers.Count == 0 ? "" : previousLayers[0].LayerName + "[0][0]");
                result += ($"{firstColumn,-firstColumnWidth}{outputShape,-secondColumnWidth}{layer.TotalParams,-thirdColumnWidth}{firstPreviousLayer,-forthColumnWidth}").TrimEnd() + Environment.NewLine;
                for (int i = 1; i < previousLayers.Count; ++i)
                {
                    result += ($"{"",-(firstColumnWidth + secondColumnWidth + thirdColumnWidth)}{previousLayers[i].LayerName + "[0][0]",-forthColumnWidth}").TrimEnd() + Environment.NewLine;
                }
                result += (layer.IsOutputLayer ? line1 : line0) + Environment.NewLine;
            }
            result += "Total params: " + TotalParams.ToString("N0", CultureInfo.InvariantCulture)+Environment.NewLine;
            result += "Trainable params: " + (TotalParams-NonTrainableParams).ToString("N0", CultureInfo.InvariantCulture) + Environment.NewLine;
            result += "Non-trainable params: " + (NonTrainableParams).ToString("N0", CultureInfo.InvariantCulture);
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
                var firstColumn = l.LayerName + " (" + l.LayerType() + ")";
                if (firstColumn.Length > firstColumnWidth - 1)
                {
                    firstColumn = firstColumn.Substring(0, firstColumnWidth - 1);
                }
                result += ($"{firstColumn,-firstColumnWidth}{outputShape,-secondColumnWidth}{l.TotalParams,-thirdColumnWidth}").TrimEnd() + Environment.NewLine;
                result += (l.IsOutputLayer ? line1 : line0) + Environment.NewLine;
            }
            result += "Total params: " + TotalParams.ToString("N0", CultureInfo.InvariantCulture) + Environment.NewLine;
            result += "Trainable params: " + (TotalParams - NonTrainableParams).ToString("N0", CultureInfo.InvariantCulture) + Environment.NewLine;
            result += "Non-trainable params: " + (NonTrainableParams).ToString("N0", CultureInfo.InvariantCulture);
            return result;
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

        /// <summary>
        /// Change the last activation layer from SoftMax to SoftMaxWithHierarchy
        /// (the output has categories with sub categories)
        /// </summary>
        /// <param name="activationParameters"></param>
        public void SetSoftmaxWithHierarchy(float[] activationParameters)
        {
            if (!Layers.Last().IsSoftmaxActivationLayer())
            {
                throw new ArgumentException("last layer must be SoftMax Layer");
            }
            var layerName = Layers.Last().LayerName;
            RemoveAndDisposeLastLayer();
            var shape = new []{activationParameters.Length};
            var tensor = MemoryPool.GetFloatTensor(shape);
            new CpuTensor<float>(shape, activationParameters).CopyTo(tensor);
            Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY, tensor, layerName);
        }
    }
}
