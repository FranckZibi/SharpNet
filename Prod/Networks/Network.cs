using System;
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
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.Layers;
using SharpNet.Models;
using SharpNet.Optimizers;
using H5GroupId = System.Int64;


namespace SharpNet.Networks
{
    public partial class Network : AbstractModel, IDisposable
    {
        #region private fields
        public readonly List<EpochData> EpochData = new();
        private Tensor _buffer;
        private readonly DateTime _timeStampCreation = DateTime.Now;
        // bytes/batchSize needed for forward & backward propagation
        #endregion

        #region public fields

        public NetworkSample NetworkSample => (NetworkSample)Sample;
        public NetworkConfig Config => NetworkSample.Config;
        public DataAugmentationSample DA => NetworkSample.DA;
        public Random Rand { get; } = new Random(0);
        public List<Layer> Layers { get; } = new List<Layer>();
        public string Description { private get; set; } = "";
        public PropagationManager PropagationManager { get; }
        public TensorMemoryPool MemoryPool { get; }
        public GPUWrapper GpuWrapper { get; }
        #endregion

        public Network(NetworkConfig config, DataAugmentationSample da) : this(new NetworkSample(new ISample[]{config, da}))
        {
        }

        public Network(NetworkSample sample, string workingDirectory, string modelName) : base(sample, workingDirectory, modelName)
        {
        }

        /// <param name="sample"></param>
            /// <param name="masterNetworkIfAny">
            ///     if the current network is a slave network doing computation for its master network:
            ///         the reference of the master network
            ///     else:
            ///         null
            /// </param>
        public Network(NetworkSample sample, Network masterNetworkIfAny = null) : this(sample, sample.Config.WorkingDirectory, sample.Config.ModelName)
        {
            //Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, Config.LogFile);
            Utils.ConfigureThreadLog4netProperties(sample.Config.WorkingDirectory, Config.ModelName);

            //a slave network will have access to only one resource (1 Cpu or 1 GPU)
            Debug.Assert(masterNetworkIfAny == null || Config.ResourceIds.Count == 1);

            _masterNetworkIfAny = masterNetworkIfAny;
            GpuWrapper = UseGPU ? GPUWrapper.FromDeviceId(Config.ResourceIds[0]) : null;
            _swComputeMetrics = new Stopwatch();
            CreateWorkingDirectoryIfNeeded();
            MemoryPool = new TensorMemoryPool(GpuWrapper);
            PropagationManager = new PropagationManager(Layers, MemoryPool, ForwardPropagationTrainingTime, ForwardPropagationInferenceTime, BackwardPropagationTime, _updateWeightsTime);
            if (IsMaster && Config.ResourceIds.Count>=2)
            {
                //we create the slave networks
                foreach(var slaveResourceId in Config.ResourceIds.Skip(1))
                {
                    Log.Debug("starting thread for network running on deviceId " + slaveResourceId);
                    new Thread(() => SlaveThread(this, slaveResourceId)).Start();
                }
                while (_slaveNetworks.Count != Config.ResourceIds.Count - 1)
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
            MemoryPool.FreeFloatTensor(ref _buffer);
            MemoryPool.FreeFloatTensor(ref _yPredictedForEpoch);
            MemoryPool.FreeFloatTensor(ref _yExpectedForEpoch);
            foreach (var t in all_x_miniBatch)
            {
                MemoryPool.FreeFloatTensor(t);
            }
            all_x_miniBatch.Clear();
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

        #region network construction: adding layers
        public Network Input(int[] shape_CHW, string layerName = "")
        {
            return Input(shape_CHW[0], shape_CHW[1], shape_CHW[2], layerName);
        }
        public Network Input(int channelCount, int h, int w, string layerName = "")
        {
            Layers.Add(new InputLayer(channelCount, h, w, this, layerName));
            return this;
        }
        public Network InputAndEmbedding(int maxWordsBySentence, int vocabularySize, int embeddingDim, int indexInLastDimensionToUse, double lambdaL2Regularization, string layerName = "")
        {
            Debug.Assert(Layers.Count == 0);
            Input(maxWordsBySentence, -1, -1);
            Embedding(vocabularySize, embeddingDim, indexInLastDimensionToUse, lambdaL2Regularization, 0, false, layerName);
            return this;
        }
        public Network Embedding(int vocabularySize, 
            int embeddingDim, 
            int indexInLastDimensionToUse, 
            double lambdaL2Regularization,
            float clipValueForGradients,
            bool divideGradientsByTimeSteps,
            string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Debug.Assert(Layers.Last().IsInputLayer);

            Layers.Add(new EmbeddingLayer(vocabularySize, embeddingDim, indexInLastDimensionToUse, lambdaL2Regularization, clipValueForGradients, divideGradientsByTimeSteps, true, this, layerName));
            return this;
        }

        public Network SwitchSecondAndThirdDimension(bool addOneDimensionInOutputShape, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            var layer = new SwitchSecondAndThirdDimensionLayer(addOneDimensionInOutputShape, this, layerName);
            Layers.Add(layer);
            return this;
        }

        public Network SimpleRNN(int hiddenSize, bool returnSequences, bool isBidirectional, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            var simpleRnn = new RecurrentLayer(hiddenSize, cudnnRNNMode_t.CUDNN_RNN_TANH, cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS, returnSequences, isBidirectional, 1, 0.0, false /* isEncoder */,-1 /* encoder layer index */, true, this, layerName);
            Layers.Add(simpleRnn);
            return this;
        }

        public Network LSTM(int hiddenSize, bool returnSequences, bool isBidirectional, int numLayers, double dropoutRate, bool isEncoder, string layerName = "")
        {
            return RecurrentLayer(hiddenSize, cudnnRNNMode_t.CUDNN_LSTM, cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS, returnSequences, isBidirectional, numLayers, dropoutRate, isEncoder, -1, layerName);
        }

        public Network GRU(int hiddenSize, bool returnSequences, bool isBidirectional, int numLayers, double dropoutRate, bool isEncoder, string layerName = "")
        {
            return RecurrentLayer(hiddenSize, cudnnRNNMode_t.CUDNN_GRU, cudnnRNNBiasMode_t.CUDNN_RNN_DOUBLE_BIAS, returnSequences, isBidirectional, numLayers, dropoutRate, isEncoder, -1, layerName);
        }

        private Network RecurrentLayer(int hiddenSize, cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode, bool returnSequences, bool isBidirectional, int numLayers, double dropoutRate, bool isEncoder, int encoderLayerIndexIfAny, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Debug.Assert(UseGPU);

            if (numLayers == 1 && dropoutRate > 1e-6)
            {
                throw new ArgumentException("invalid dropoutRate (" + dropoutRate + ") for a 1 Layer RecurrentLayer");
            }

            if (numLayers >= 2)
            {
                //there is ann issue with cuDNN 8.* when using multi layer RNN with dropout.
                //We are using a workaround: building several stacked single layer RNN with a dropout layer between each of them
                for (int i = 0; i < numLayers-1; ++i)
                {
                    /* first 'numLayers-1' layers: single layer with no dropout, always returning a full sequence */
                    RecurrentLayer(hiddenSize, cellMode, biasMode, true, isBidirectional, 1, 0.0, false, encoderLayerIndexIfAny, "");
                    Dropout(dropoutRate);
                    encoderLayerIndexIfAny = -1;
                }
                /* last layer: single layer with no dropout, returning a full sequence iif 'returnSequences' is true */
                RecurrentLayer(hiddenSize, cellMode, biasMode, returnSequences, isBidirectional, 1, 0.0, isEncoder, encoderLayerIndexIfAny, "");
            }
            else
            {
                var recurrentLayer = new RecurrentLayer(hiddenSize, cellMode, biasMode, returnSequences, isBidirectional, numLayers, dropoutRate, isEncoder, encoderLayerIndexIfAny, true, this, layerName);
                Layers.Add(recurrentLayer);
            }

            return this;
        }

        public Network DecoderLayer(int encoderLayerIndex, int numLayers, double dropoutRate, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Debug.Assert(UseGPU);

            /* we look for the associate encoder */
            var layer = Layers[encoderLayerIndex];
            while (! (layer is RecurrentLayer))
            {
                layer = layer.PreviousLayers[0];
            }
            var encoder  = (RecurrentLayer)layer;

            var xLayer = Layers.Last();
            var xLayerShape = xLayer.OutputShape(1);
            Debug.Assert(xLayerShape.Length == 3);
            var timeSteps = xLayerShape[1];
            bool returnSequences = timeSteps != 1;

            return RecurrentLayer(encoder.HiddenSize, encoder.CellMode, encoder.BiasMode, returnSequences,
                encoder.IsBidirectional, numLayers, dropoutRate, false, encoderLayerIndex, layerName);
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
        public Network Dropout(double dropoutRate, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new DropoutLayer(dropoutRate, this, layerName));
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
                MemoryPool.GetFloatTensor(ref _buffer, new[] { yExpectedMiniBatch.Shape[0] });
                var blockLoss = yExpectedMiniBatch.ComputeLoss(yPredictedMiniBatch, Config.LossFunction, _buffer);
                learningRateFinder.AddLossForLastBlockId(blockLoss);
            }
            MiniBatchGradientDescentForSingleEpoch(trainingDataSet, miniBatchSizeForAllWorkers, learningRateFinder, CallBackAfterEachMiniBatch);
            var fileName = Path.Combine(WorkingDirectory, DynamicModelName + "_LearningRateFinder.csv");
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
        /// <param name="trainingDataset"></param>
        /// <param name="validationDatasetIfAny"></param>
        public override void Fit(IDataSet trainingDataset, IDataSet validationDatasetIfAny)
        {
            int miniBatchSizeForAllWorkers = Config.BatchSize;
            int numEpochs = Config.NumEpochs;
            var learningRateComputer = Config.GetLearningRateComputer();

            try
            {
                Debug.Assert(Config.TypeSize == trainingDataset.TypeSize);
                Debug.Assert(miniBatchSizeForAllWorkers >= 1);
                _spInternalFit.Start();
                StartTimer("Fit_Prepare", ForwardPropagationTrainingTime);

                FreezeSelectedLayers();

                Log.Debug("Fit( " /*+ Tensor.ShapeToString(XMiniBatch_Shape(trainingDataSet.Count))*/ +" => " + Tensor.ShapeToString(trainingDataset.YMiniBatch_Shape(trainingDataset.Count))+" )");
                Log.Info(ToString());


                if (UseGPU)
                {
                    Log.Debug(GpuWrapper.ToString());
                }
                Log.Debug("Training dataset: " + trainingDataset);
                if (validationDatasetIfAny != null)
                {
                    Log.Debug("Validation dataset: " + validationDatasetIfAny);
                }
                Log.Info("#Epochs=" + numEpochs + " BatchSize=" + miniBatchSizeForAllWorkers+" Name="+Description);
                if (Config.DisplayTensorContentStats)
                {
                    Log.Debug("Initial Tensor Content stats" + Environment.NewLine + ContentStats() + Environment.NewLine);
                }

                //Info(GpuWrapper.ToString());

                StopTimer("Fit_Prepare", ForwardPropagationTrainingTime);


                var lastAutoSaveTime = DateTime.Now; //last time we saved the network
                IDictionary<MetricEnum, double> validationMetrics = null;
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
                    var yPredicted = MiniBatchGradientDescentForSingleEpoch(trainingDataset, miniBatchSizeForAllWorkers, learningRateComputer, null);
                    #endregion

                    //We display stats about the just finished epoch
                    if (Config.DisplayTensorContentStats)
                    {
                        Log.Debug("End of Epoch:" + epoch + " Tensor Content stats" + Environment.NewLine+ContentStats()+Environment.NewLine);
                    }

                    StartTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);
                    var trainingMetrics = ComputeMetrics(_yExpectedForEpoch, yPredicted);
                    var lossAndAccuracyMsg = IModel.MetricsToString(trainingMetrics, "");
                    if (validationDatasetIfAny != null)
                    {
                        //We compute the validation loss&accuracy
                        if (ShouldUseFullTestDataSetForLossAndAccuracy(learningRateComputer, epoch, numEpochs))
                        {
                            validationMetrics = ComputeMetricsForTestDataSet(miniBatchSizeForAllWorkers, validationDatasetIfAny);
                            lossAndAccuracyMsg += " - " + IModel.MetricsToString(validationMetrics, "val_");
                        }
                        else
                        {
                            var percentageToUseInTestDataSet = validationDatasetIfAny.PercentageToUseForLossAndAccuracyFastEstimate;
                            //we'll compute loss and accuracy using only 'percentageToUsForLossAndAccuracyFastEstimate' of the test data set
                            if (percentageToUseInTestDataSet > 1e-6)
                            { 
                                using var subDataSet = validationDatasetIfAny.SubDataSet(percentageToUseInTestDataSet);
                                validationMetrics = ComputeMetricsForTestDataSet(miniBatchSizeForAllWorkers, subDataSet);
                                lossAndAccuracyMsg += " - " + IModel.MetricsToString(validationMetrics, "estimate_val_");
                            }
                        }

                    }
                    StopTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);

                    double secondsForEpoch = swEpoch.Elapsed.TotalSeconds;
                    double nbStepsByEpoch = ((double)trainingDataset.Count) / miniBatchSizeForAllWorkers;
                    var msByStep = (1000 * secondsForEpoch) / nbStepsByEpoch;
                    Log.Info("Epoch " + epoch + "/" + numEpochs + " - " + Math.Round(secondsForEpoch, 0) + "s " + Math.Round(msByStep, 0) + "ms/step - lr: "+Math.Round(learningRateAtEpochStart, 8)+" - "+lossAndAccuracyMsg+" - "+ Description);
                    Log.Debug(MemoryInfo());
                    //if it is the last epoch, we'll save Layer KPI
                    if (epoch == numEpochs)
                    {
                        Log.Debug(LayersKpi());
                    }

                    #region we save stats about the just finished epoch
                    var currentEpochData = new EpochData(epoch, learningRateAtEpochStart, lrMultiplicativeFactorFromReduceLrOnPlateau, secondsForEpoch, trainingMetrics, validationMetrics);
                    EpochData.Add(currentEpochData);
                    #endregion

                    #region we save the network in a file if necessary
                    if (   //if we have finished training
                           ((epoch == numEpochs) && (numEpochs > 10))
                            //or if we should save the network every 'Config.AutoSaveIntervalInMinutes' minutes
                        || ( (Config.AutoSaveIntervalInMinutes>=0) && (DateTime.Now - lastAutoSaveTime).TotalMinutes > Config.AutoSaveIntervalInMinutes)
                        || learningRateComputer.ShouldCreateSnapshotForEpoch(epoch)
                        || trainingDataset.ShouldCreateSnapshotForEpoch(epoch, this)
                        || ShouldStopTrainingBecauseOfEarlyStopping(EpochData, Config.EarlyStoppingRounds)
                    )
                    {
                        trainingDataset.Save(this, WorkingDirectory, DynamicModelName);
                        lastAutoSaveTime = DateTime.Now;
                    }
                    #endregion

                    if (Config.SaveNetworkStatsAfterEachEpoch)
                    {
                        var networkStatFileName = Path.Combine(WorkingDirectory, DynamicModelName + "_NetworkStats.txt");
                        Log.Info("Saving network '" + Description + "' stats in " + networkStatFileName);
                        File.WriteAllText(networkStatFileName, ContentStats());
                    }
                    if (ShouldStopTrainingBecauseOfEarlyStopping(EpochData, Config.EarlyStoppingRounds))
                    {
                        Log.Info("Stopping Training because of EarlyStopping");
                        break;
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
                    var testsCsv = string.IsNullOrEmpty(trainingDataset.Name)?"Tests.csv": ("Tests_"+ trainingDataset.Name + ".csv");
                    File.AppendAllText(Utils.ConcatenatePathWithFileName(WorkingDirectory, testsCsv), line);
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


        /// <summary>
        /// if the validation loss has degraded for several consecutive epochs  (at least 'earlyStoppingRounds')
        /// then we stop the training
        /// </summary>
        /// <returns></returns>
        private static bool ShouldStopTrainingBecauseOfEarlyStopping(List<EpochData> epochData, int earlyStoppingRounds)
        {
            if (earlyStoppingRounds <= 0)
            {
                return false;
            }
            // number of consecutive epochs where the validation loss has degraded.
            int nbConsecutiveEpochsWithDegradationOfValidationLoss = 0;
            for (int i = epochData.Count - 1; i >= 1; --i)
            {
                var currentValidationLoss = epochData[i].ValidationLoss;
                var previousValidationLoss = epochData[i - 1].ValidationLoss;
                if (double.IsNaN(previousValidationLoss) || double.IsNaN(currentValidationLoss))
                {
                    break;
                }
                var validationLossHasDegraded = currentValidationLoss > previousValidationLoss;
                if (!validationLossHasDegraded)
                {
                    break;
                }
                ++nbConsecutiveEpochsWithDegradationOfValidationLoss;
            }
            return nbConsecutiveEpochsWithDegradationOfValidationLoss >= earlyStoppingRounds;
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
        public IDictionary<MetricEnum, double> ComputeMetricsForTestDataSet(int miniBatchSize, IDataSet testDataSet)
        {
            //We perform a mini batch gradient descent in Testing mode:
            //  there will be no shuffling/data augmentation.
            var yPredicted = MiniBatchGradientDescentForSingleEpoch(testDataSet, miniBatchSize);
            return ComputeMetrics(testDataSet.Y, yPredicted);
        }
        private IDictionary<MetricEnum, double> ComputeMetrics(Tensor yExpected, Tensor yPredicted)
        {
            _swComputeMetrics?.Start();
            var result = new Dictionary<MetricEnum, double>();
            yExpected = ReformatToCorrectDevice_GPU_or_CPU(yExpected);
            yPredicted = ReformatToCorrectDevice_GPU_or_CPU(yPredicted);
            var batchSize = yExpected.Shape[0];
            MemoryPool.GetFloatTensor(ref _buffer, new[] { batchSize });
            foreach (var metric in Config.Metrics)
            {
                result[metric] = yExpected.ComputeMetric(yPredicted, metric, Config.LossFunction, _buffer);
            }
            _swComputeMetrics?.Stop();
            return result;
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
            return Predict(new List<Tensor> {X}, isTraining);
        }

        private Tensor Predict(List<Tensor> allX, bool isTraining)
        {
            var batchSize = allX[0].Shape[0];
            var yPredicted = MemoryPool.GetFloatTensor(Layers.Last().OutputShape(batchSize));
            for(int i= 0;i<allX.Count;++i)
            {
                allX[i] = ReformatToCorrectDevice_GPU_or_CPU(allX[i]);
            }
            PropagationManager.Forward(allX, yPredicted, isTraining);
            return yPredicted;
        }

        public override CpuTensor<float> Predict(IDataSet dataset)
        {
            return Predict(dataset, Config.BatchSize);
        }

        public override int GetNumEpochs()
        {
            throw new NotImplementedException();
        }

        public override string GetDeviceName()
        {
            throw new NotImplementedException();
        }

        public override double GetLearningRate()
        {
            throw new NotImplementedException();
        }


        public CpuTensor<float> Predict(IDataSet dataset, int miniBatchSizeForAllWorkers)
        {
            var yPredicted = MiniBatchGradientDescentForSingleEpoch(dataset, miniBatchSizeForAllWorkers);
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

            if ( (epoch >= 2||dataSet is ITimeSeriesDataSet) && Config.RandomizeOrder && isTraining)
            {
                Utils.Shuffle(shuffledElementId, Rand);
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

            var all_x_miniBatch_cpu_allWorkers = new List<CpuTensor<float>>();
            var shapeForFirstLayer = Layers[0].OutputShape(miniBatchSizeForAllWorkers);
            foreach (var xShape in dataSet.XMiniBatch_Shape(shapeForFirstLayer))
            {
                all_x_miniBatch_cpu_allWorkers.Add(new CpuTensor<float>(xShape));
            }
            var yExpected_miniBatch_cpu_allWorkers = new CpuTensor<float>(dataSet.YMiniBatch_Shape(miniBatchSizeForAllWorkers), null);
            var shuffledElementIdMemory = new Memory<int>(shuffledElementId);
            for (int firstIndexInShuffledElementId = 0; firstIndexInShuffledElementId < dataSet.Count; )
            {
                //Log.Info("Processing epoch " + epoch + " for elements [" + firstIndexInShuffledElementId + ":]");

                //we initialize miniBatch input (xMiniBatch) and expected output (yExpectedMiniBatchCpu)
                StartTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                bool withDataAugmentation = DA.UseDataAugmentation && (epoch >= 2) && isTraining;
                int actualNumberOfLoadedItems = dataSet.LoadMiniBatch(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementId, DA, all_x_miniBatch_cpu_allWorkers, yExpected_miniBatch_cpu_allWorkers);
                StopTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                //we copy yExpected_miniBatch_cpu_allWorkers from CPU to appropriate target (CPU or GPU)
                var yExpectedForMiniBatch_allWorkers = _yExpectedForEpoch.RowSlice(firstIndexInShuffledElementId, miniBatchSizeForAllWorkers);
                yExpected_miniBatch_cpu_allWorkers.CopyTo(yExpectedForMiniBatch_allWorkers);

                //we launch the forward & backward computation on all slave networks
                var usedSlaves = new List<Network>();
                int firstIndexInShuffledElement_slave = firstIndexInShuffledElementId + miniBatchSizeForEachWorker;
                foreach (var slave in _slaveNetworks)
                {
                    var x_miniBatch_cpu_slave = Tensor.RowSlice(all_x_miniBatch_cpu_allWorkers, firstIndexInShuffledElement_slave - firstIndexInShuffledElementId, miniBatchSizeForEachWorker);
                    var yExpected_miniBatch_cpu_slave = yExpected_miniBatch_cpu_allWorkers.RowSlice(firstIndexInShuffledElement_slave - firstIndexInShuffledElementId, miniBatchSizeForEachWorker);
                    var yPredicted_miniBatch_slave = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElement_slave, miniBatchSizeForEachWorker);
                    slave._slaveParamForMiniBatchGradientDescent = Tuple.Create(x_miniBatch_cpu_slave, yExpected_miniBatch_cpu_slave, yPredicted_miniBatch_slave, isTraining);
                    slave._slaveStatus = SLAVE_NETWORK_STATUS.PERFORM_FORWARD_AND_BACKWARD_PROPAGATION;
                    firstIndexInShuffledElement_slave += miniBatchSizeForEachWorker;
                    usedSlaves.Add(slave);
                }

                //we launch the forward & backward propagation on the master network
                var all_x_miniBatch_cpu_master = Tensor.RowSlice(all_x_miniBatch_cpu_allWorkers, 0, miniBatchSizeForEachWorker);
                for (int x = 0; x < all_x_miniBatch_cpu_master.Count; ++x)
                {
                    if (all_x_miniBatch.Count <= x)
                    {
                        all_x_miniBatch.Add(MemoryPool.GetFloatTensor(all_x_miniBatch_cpu_master[x].Shape));
                    }
                    else
                    {
                        var tmp_x_miniBatch = all_x_miniBatch[x];
                        MemoryPool.GetFloatTensor(ref tmp_x_miniBatch, all_x_miniBatch_cpu_master[x].Shape);
                    }
                    all_x_miniBatch_cpu_master[x].CopyTo(all_x_miniBatch[x]);
                }

                var yPredicted_miniBatch_master = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElementId, miniBatchSizeForEachWorker);
                var yExpected_miniBatch_master = _yExpectedForEpoch.RowSlice(firstIndexInShuffledElementId, miniBatchSizeForEachWorker);

                PropagationManager.Forward(all_x_miniBatch, yPredicted_miniBatch_master, isTraining);
                if (isTraining)
                {
                    PropagationManager.Backward(yExpected_miniBatch_master, yPredicted_miniBatch_master, Config.LossFunction);
                }

                if (_slaveNetworks.Any())
                { 
                    //we wait for all slave to finish the forward & backward propagation pass
                    StartTimer("WaitForSlave_Forward", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                    WaitForAllSlavesInStatus(SLAVE_NETWORK_STATUS.IDLE);
                    StopTimer("WaitForSlave_Forward", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                }

                if (isTraining)
                {
                    foreach (var usedSlave in usedSlaves)
                    {
                        StartTimer("CopyGradients", BackwardPropagationTime);
                        AddGradientFromSlaveNetwork(usedSlave);
                        StopTimer("CopyGradients", BackwardPropagationTime);
                    }
                    double percentagePerformedInEpoch = firstIndexInShuffledElementId / (double) dataSetCountWithExtraBufferAtEnd;
                    var learningRate = learningRateComputerIfTraining.LearningRate(epoch, percentagePerformedInEpoch, lrMultiplicativeFactorFromReduceLrOnPlateau);
                    var maxLearningRate = learningRateComputerIfTraining.MaxLearningRate;
                    PropagationManager.UpdateWeights(miniBatchSizeForAllWorkers, learningRate, maxLearningRate);
                }

                if (!isTraining && dataSet is ITimeSeriesDataSet set)
                {
                    StartTimer("SetBatchPredictions", ForwardPropagationInferenceTime);
                    //During inference for TimeSeries, we'll need the previous predicted values to predict the next one
                    var batchPredictions = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElementId, actualNumberOfLoadedItems);
                    var batchElementIds = shuffledElementIdMemory.Slice(firstIndexInShuffledElementId, actualNumberOfLoadedItems).ToArray();
                    set.SetBatchPredictionsForInference(batchElementIds, batchPredictions);
                    StopTimer("SetBatchPredictions", ForwardPropagationInferenceTime);
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

            all_x_miniBatch_cpu_allWorkers.ForEach(t => t.Dispose());
            all_x_miniBatch_cpu_allWorkers.Clear();
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


        public bool UseGPU => Config.ResourceIds.Max() >= 0;
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
        public bool CurrentEpochIsAbsolutelyBestInValidationLoss()
        {
            return
                EpochData.Count >= 1
                && EpochData.All(e => !double.IsNaN(e.ValidationLoss))
                && Math.Abs(EpochData.Select(e => e.ValidationLoss).Min() - EpochData.Last().ValidationLoss) < 1e-6;
        }


        private static string ComputeModelName(string description)
        {
            var desc = (string.IsNullOrEmpty(description) ? "Network" : Utils.ToValidFileName(description));
            var timeStamp = DateTime.Now.ToString("yyyyMMdd_HHmm", CultureInfo.InvariantCulture);
            return desc + "_" + timeStamp + "_" + Thread.CurrentThread.ManagedThreadId;
        }
        public string DynamicModelName
        {
            get
            {
                var desc = (string.IsNullOrEmpty(Description) ? "Network" : Utils.ToValidFileName(Description));
                var epoch = EpochData.Count;
                var trainingLoss = "";
                if (epoch >= 1 && !double.IsNaN(EpochData.Last().TrainingLoss))
                {
                    trainingLoss = "_" + Math.Round(EpochData.Last().TrainingLoss, 4).ToString(CultureInfo.InvariantCulture).Replace(".", "_");
                }
                var validationLoss = "";
                if (epoch >=1 && !double.IsNaN(EpochData.Last().ValidationLoss))
                {
                    validationLoss = "_" + Math.Round(EpochData.Last().ValidationLoss, 4).ToString(CultureInfo.InvariantCulture) .Replace(".", "_");
                }
                var timeStamp = _timeStampCreation.ToString("yyyyMMdd_HHmm", CultureInfo.InvariantCulture);
                return desc + "_" + epoch + trainingLoss + validationLoss + "_" + timeStamp + "_" + Thread.CurrentThread.ManagedThreadId;
            }
        }
        private void CreateWorkingDirectoryIfNeeded()
        {
            if (!string.IsNullOrEmpty(WorkingDirectory) && !Directory.Exists(WorkingDirectory))
            {
                Directory.CreateDirectory(WorkingDirectory);
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
