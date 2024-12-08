using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
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
using SharpNet.Models;
using SharpNet.Optimizers;


namespace SharpNet.Networks
{
    public partial class Network : Model
    {
        #region private fields
        public readonly List<EpochData> EpochData = new();
        private Tensor _buffer;
        //private readonly DateTime _timeStampCreation = DateTime.Now;
        // bytes/batchSize needed for forward & backward propagation
        #endregion

        #region public fields
        // ReSharper disable once MemberCanBePrivate.Global
        public NetworkSample NetworkSample => (NetworkSample)ModelSample;
        public NetworkSample Sample => NetworkSample;
        public Random Rand { get; } = new (0);
        public List<Layer> Layers { get; } = new ();
        public PropagationManager PropagationManager { get; }
        public TensorMemoryPool MemoryPool { get; }
        public GPUWrapper GpuWrapper { get; }
        #endregion

        

        /// <param name="modelSample"></param>
        /// <param name="modelName"></param>
        /// <param name="buildLayers"></param>
        /// <param name="masterNetworkIfAny">
        ///     if the current network is a slave network doing computation for its master network:
        ///         the reference of the master network
        ///     else:
        ///         null
        /// </param>
        /// <param name="datasetSample"></param>
        /// <param name="workingDirectory"></param>
        public Network(NetworkSample modelSample, AbstractDatasetSample datasetSample, string workingDirectory, string modelName, bool buildLayers, Network masterNetworkIfAny = null) : base(modelSample, workingDirectory, modelName)
        {
            //Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, Config.LogFile);
            //a slave network will have access to only one resource (1 Cpu or 1 GPU)
            Debug.Assert(masterNetworkIfAny == null || Sample.ResourceIds.Count == 1);

            MasterNetworkIfAny = masterNetworkIfAny;
            GpuWrapper = UseGPU ? GPUWrapper.FromDeviceId(Sample.ResourceIds[0]) : null;
            _swComputeMetrics = new Stopwatch();
            CreateWorkingDirectoryIfNeeded();
            MemoryPool = new TensorMemoryPool(GpuWrapper);
            PropagationManager = new PropagationManager(Layers, MemoryPool, ForwardPropagationTrainingTime, ForwardPropagationInferenceTime, BackwardPropagationTime, _updateWeightsTime);
            if (buildLayers)
            {
                Sample.BuildLayers(this, datasetSample);
            }
            if (IsMaster && Sample.ResourceIds.Count >= 2)
            {
                //we create the slave networks
                foreach (var slaveResourceId in Sample.ResourceIds.Skip(1))
                {
                    LogDebug("starting thread for network running on deviceId " + slaveResourceId);
                    new Thread(() => SlaveThread(this, buildLayers, datasetSample, slaveResourceId)).Start();
                }
                while (_slaveNetworks.Count != Sample.ResourceIds.Count - 1)
                {
                    Thread.Sleep(1);
                }
            }
        }

        public override void Dispose()
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
            EpochData.Clear();
            MemoryPool.FreeFloatTensor(ref _buffer);
            MemoryPool.FreeFloatTensor(ref _yPredictedForEpoch);
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
                LogDebug("After clearing memory: " + MemoryInfo());
            }
        }

        #region network construction: adding layers
        public Network Input(int[] shape_CHW, string layerName = "")
        {
            return Input(shape_CHW[0], shape_CHW.Length>=2?shape_CHW[1]:-1, shape_CHW.Length>=3?shape_CHW[2]:-1, layerName);
        }

        public Network Input_and_Embedding_if_required(AbstractDatasetSample datasetSample, int embeddingDim, double lambdaL2Regularization, float clipValueForGradients= 0, string layerName = "")
        {
            Input(datasetSample.X_Shape(1).Skip(1).ToArray(), layerName);
            var (vocabularySizes, embeddingDims, indexesInLastDimensionToUse, embeddingTensorIndex) = datasetSample.EmbeddingDescription(embeddingDim);
            if (indexesInLastDimensionToUse.Length != 0)
            {
                //We need to add an embedding layer to manage categorical features
                Embedding(vocabularySizes, embeddingDims, indexesInLastDimensionToUse, embeddingTensorIndex, lambdaL2Regularization, clipValueForGradients);
            }
            return this;
        }

        public Network Input(int channelCount, int h, int w, string layerName = "")
        {
            Layers.Add(new InputLayer(channelCount, h, w, this, layerName));
            return this;
        }

        public Network Reshape(int channelCount, int h, int w, string layerName = "")
        {
            Layers.Add(new ReshapeLayer(channelCount, h, w, this, layerName));
            return this;
        }

        public Network Embedding(int[] vocabularySizes,
                int[] embeddingDims,
                int[] indexesInLastDimensionToUse,
                int[] embeddingTensorIndex,
                double lambdaL2Regularization,
                float clipValueForGradients = 0f,

                bool divideGradientsByTimeSteps = false,
                //bool divideGradientsByTimeSteps = true,

                string layerName = "")
            {
            Debug.Assert(Layers.Count >= 1);
            Debug.Assert(Layers.Last().IsInputLayer);

            Layers.Add(new EmbeddingLayer(EmbeddingLayer.ToEmbeddingLayerDescription(vocabularySizes, embeddingDims, indexesInLastDimensionToUse, embeddingTensorIndex), lambdaL2Regularization, clipValueForGradients, divideGradientsByTimeSteps, true, this, layerName));
            return this;
        }
        public Network SwitchSecondAndThirdDimension(bool addOneDimensionInOutputShape, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            var layer = new SwitchSecondAndThirdDimensionLayer(addOneDimensionInOutputShape, this, layerName);
            Layers.Add(layer);
            return this;
        }

        public Network SimpleRNN(int hiddenSize, bool returnSequences, bool isBidirectional, int numLayers = 1, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            var biasMode = Sample.CompatibilityMode == NetworkSample.CompatibilityModeEnum.PyTorch? cudnnRNNBiasMode_t.CUDNN_RNN_DOUBLE_BIAS: cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS;
            var simpleRnn = new RecurrentLayer(hiddenSize, cudnnRNNMode_t.CUDNN_RNN_TANH, biasMode, returnSequences, isBidirectional, numLayers, 0.0, false /* isEncoder */,-1 /* encoder layer index */, true, this, layerName);
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

        public Network PositionalEncodingAttnIsAllYouNeedLayer(int N, string layerName = "")
        {
            var linearFunctionLayer = new PositionalEncodingAttnIsAllYouNeedLayer(N, this, layerName);
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

        // ReSharper disable once UnusedMember.Global
        public Network ScaledDotProductAttention(bool use_scale, bool is_causal = false, string layerName = "")
        {
            var previousLayerIndex = Layers.Count - 1;
            int queriesLayerIndex = previousLayerIndex;
            int keysLayerIndex = previousLayerIndex;
            int valuesLayerIndex = previousLayerIndex;
            return ScaledDotProductAttention(use_scale, is_causal, queriesLayerIndex, keysLayerIndex, valuesLayerIndex, layerName);
        }
        public Network ScaledDotProductAttention(bool use_scale, bool is_causal, int queriesLayerIndex, int keysLayerIndex, int valuesLayerIndex, string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ScaledDotProductAttentionLayer(use_scale, is_causal, queriesLayerIndex, keysLayerIndex, valuesLayerIndex, this, layerName));
            return this;
        }

        public Network MultiHeadAttention(int num_heads, int key_dim, int value_dim, bool use_bias_K_V_K, bool use_bias_O,
            bool is_causal, int queriesLayerIndex, int keysLayerIndex, int valuesLayerIndex,
            string layerName = "")
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new MultiheadAttention(num_heads, key_dim, value_dim, use_bias_K_V_K, use_bias_O, is_causal, queriesLayerIndex, keysLayerIndex, valuesLayerIndex, this, layerName));
            return this;
        }

        public Network Conv1D(int filtersCount, int kernelWidth, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, string layerName = "")
        {
            return Conv1D(filtersCount, kernelWidth, stride, paddingType, lambdaL2Regularization, useBias, Layers.Count - 1, layerName);
        }
        public Network Conv1D(int filtersCount, int kernelWidth, int stride, ConvolutionLayer.PADDING_TYPE paddingType, double lambdaL2Regularization, bool useBias, int previousLayerIndex, string layerName = "")
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
            if (dropoutRate > 0)
            {
                Layers.Add(new DropoutLayer(dropoutRate, this, layerName));
            }
            return this;
        }
        public Network Activation(cudnnActivationMode_t activationFunction, string layerName = "", int lastLayerIndex = -1)
        {
            return Activation(activationFunction, null, layerName, lastLayerIndex);
        }
        public Network Activation(cudnnActivationMode_t activationFunction, Tensor activationParameter, string layerName = "", int lastLayerIndex = -1)
        {
            Debug.Assert(Layers.Count >= 1);
            Layers.Add(new ActivationLayer(activationFunction, activationParameter, this, layerName, lastLayerIndex));
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
            return MaxPooling(-1, -1, -1, -1, previousLayerIndex, layerName);
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

        public Network LayerNorm(int last_D_dimension = 1, double epsilon = SharpNet.Layers.LayerNorm.DEFAULT_EPSILON, string layerName = "", int lastLayerIndex = -1)
        {
            Debug.Assert(Layers.Count >= 1);
            Debug.Assert(last_D_dimension >= 1);
            Layers.Add(new LayerNorm(last_D_dimension, epsilon, true, this, layerName, lastLayerIndex));
            return this;
        }

        public Network RMSNorm(int last_D_dimension = 1, double epsilon = SharpNet.Layers.RMSNorm.DEFAULT_EPSILON, string layerName = "", int lastLayerIndex = -1)
        {
            Debug.Assert(Layers.Count >= 1);
            Debug.Assert(last_D_dimension >= 1);
            Layers.Add(new RMSNorm(last_D_dimension, epsilon, true, this, layerName, lastLayerIndex));
            return this;
        }

        public Network Flatten(bool flattenInputTensorOnLastDimension = false)
        {
            Debug.Assert(Layers.Count >= 1);
            var flattenLayer = new FlattenLayer(flattenInputTensorOnLastDimension, this);
            Layers.Add(flattenLayer);
            return this;
        }
        #endregion

        public override string ToString()
        {
            var res = Summary();
            res += ToPytorchModule(256);
            return res;
            //return Summary();
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
            var shape = new[] { activationParameters.Length };
            var tensor = MemoryPool.GetFloatTensor(shape);
            new CpuTensor<float>(shape, activationParameters).CopyTo(tensor);
            Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY, tensor, layerName);
        }

        private int NonTrainableParams => Layers.Select(l => l.NonTrainableParams).Sum();
        // ReSharper disable once UnusedMethodReturnValue.Global
        public double FindBestLearningRate(DataSet trainingDataSet, double minLearningRate, double maxLearningRate, int miniBatchSizeForAllWorkers)
        {
            Debug.Assert(minLearningRate >= 0);
            Debug.Assert(maxLearningRate >= 0);
            Debug.Assert(maxLearningRate > minLearningRate);
            Debug.Assert(miniBatchSizeForAllWorkers >= 1);

            LogInfo(ToString());
            LogInfo("Looking for best learning rate...");
            ResetWeights(); //restore weights to their original values
            LogDebug("BatchSize: "+ miniBatchSizeForAllWorkers);
            var learningRateFinder = new LearningRateFinder(miniBatchSizeForAllWorkers, trainingDataSet.Count, minLearningRate, maxLearningRate);

            void CallBackAfterEachMiniBatch(Tensor yExpectedMiniBatch, Tensor yPredictedMiniBatch)
            {
                MemoryPool.GetFloatTensor(ref _buffer, new[] { yExpectedMiniBatch.Shape[0] });
                var blockLoss = _buffer.ComputeEvaluationMetric(yExpectedMiniBatch, yPredictedMiniBatch, Sample.GetLoss(), Sample);
                learningRateFinder.AddLossForLastBlockId(blockLoss);
            }
            MiniBatchGradientDescentForSingleEpoch(trainingDataSet, miniBatchSizeForAllWorkers, learningRateFinder, CallBackAfterEachMiniBatch, returnPredictionsForFullDataset: false, computeMetricsForFullDataset: false);
            var fileName = Path.Combine(WorkingDirectory, ModelName + "_LearningRateFinder.csv");
            File.WriteAllText(fileName, learningRateFinder.AsCsv());
            LogInfo("Stats stored in: " + fileName);
            var bestLearningRate = learningRateFinder.BestLearningRate();
            LogInfo("Best learning rate: "+ bestLearningRate+ " (with batch size="+miniBatchSizeForAllWorkers+")");
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
        [SuppressMessage("ReSharper", "ExpressionIsAlwaysNull")]
        public override (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat, IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainRankingMetricIfAvailable, IScore validationRankingMetricIfAvailable)
            Fit(DataSet trainingDataset, DataSet validationDatasetIfAny, Func<bool, bool, DataSet, DataSet, string, List<string>> save = null)
        {
            if (save == null)
            {
                save = (_, _, _, _, modelNameForEpoch) => Save(WorkingDirectory, modelNameForEpoch);
            }
            
            //FindBestLearningRate(trainingDataset, 1e-8, 10, Sample.BatchSize);

            if (ModelSample.GetLoss() == EvaluationMetricEnum.DEFAULT_VALUE)
            {
                throw new ArgumentException("Loss Function not set");
            }
            int miniBatchSizeForAllWorkers = Sample.BatchSize;
            var learningRateComputer = Sample.GetLearningRateComputer();

            try
            {
                Debug.Assert(Sample.TypeSize == trainingDataset.TypeSize);
                Debug.Assert(miniBatchSizeForAllWorkers >= 1);
                _spInternalFit.Start();
                StartTimer("Fit_Prepare", ForwardPropagationTrainingTime);
                FreezeSelectedLayers();
                LogInfo(ToString());

                if (UseGPU)
                {
                    LogDebug(GpuWrapper.ToString());
                }
                LogDebug("Training dataset: " + trainingDataset);
                if (validationDatasetIfAny != null)
                {
                    LogDebug("Validation dataset: " + validationDatasetIfAny);
                }
                LogInfo("#Epochs=" + Sample.num_epochs + " BatchSize=" + miniBatchSizeForAllWorkers+" Name="+ModelName);
                if (Sample.DisplayTensorContentStats)
                {
                    LogDebug("Initial Tensor Content stats" + Environment.NewLine + ContentStats() + Environment.NewLine);
                }

                //Info(GpuWrapper.ToString());

                StopTimer("Fit_Prepare", ForwardPropagationTrainingTime);



                //each time we save the networks, we update this dictionary with:
                //  - the timestamp when we saved the network (key of the dictionary)
                //  - the file used to save the network
                Dictionary<DateTime, List<string>> savedNetworks = new();

                for (;;)
                {
                    int epoch = EpochData.Count + 1;
                    if (epoch > Sample.num_epochs)
                    {
                        break;
                    }

                    var swEpoch = Stopwatch.StartNew();

                    var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputer.MultiplicativeFactorFromReduceLrOnPlateau(EpochData, Sample.GetLoss());
                    if (learningRateComputer.ShouldReduceLrOnPlateau(EpochData, Sample.GetLoss()))
                    {
                        LogInfo("Reducing learningRate because of plateau at epoch " + epoch + " (new multiplicative coeff:" + lrMultiplicativeFactorFromReduceLrOnPlateau + ")");
                    }

                    #region Mini Batch gradient descent
                    var learningRateAtEpochStart = learningRateComputer.LearningRate(epoch, 0, lrMultiplicativeFactorFromReduceLrOnPlateau);
                    var (_, trainingMetrics) = MiniBatchGradientDescentForSingleEpoch(trainingDataset, miniBatchSizeForAllWorkers, learningRateComputer, null, returnPredictionsForFullDataset: false, computeMetricsForFullDataset: true);
                    #endregion

                    //We display stats about the just finished epoch
                    if (Sample.DisplayTensorContentStats)
                    {
                        LogDebug("End of Epoch:" + epoch + " Tensor Content stats" + Environment.NewLine + ContentStats() + Environment.NewLine);
                    }

                    StartTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);
                    var lossAndAccuracyMsg = MetricsToString(trainingMetrics, "");
                    List<KeyValuePair<EvaluationMetricEnum, double>> validationMetrics = null;
                    if (validationDatasetIfAny != null)
                    {
                        //We compute the validation loss&accuracy
                        if (ShouldUseFullTestDataSetForLossAndAccuracy(learningRateComputer, epoch))
                        {
                            validationMetrics = ComputeMetricsForValidationDataSet(miniBatchSizeForAllWorkers, validationDatasetIfAny);
                            lossAndAccuracyMsg += " - " + MetricsToString(validationMetrics, "val_");
                        }
                        else
                        {
                            var percentageToUseInTestDataSet = validationDatasetIfAny.PercentageToUseForLossAndAccuracyFastEstimate;
                            //we'll compute loss and accuracy using only 'percentageToUsForLossAndAccuracyFastEstimate' of the test data set
                            if (percentageToUseInTestDataSet > 1e-6)
                            {
                                using var subDataSet = validationDatasetIfAny.SubDataSet(percentageToUseInTestDataSet);
                                validationMetrics = ComputeMetricsForValidationDataSet(miniBatchSizeForAllWorkers, subDataSet);
                                lossAndAccuracyMsg += " - " + MetricsToString(validationMetrics, "estimate_val_");
                            }
                        }
                    }
                    StopTimer("Fit_LossAndAccuracy", ForwardPropagationTrainingTime);

                    double secondsForEpoch = swEpoch.Elapsed.TotalSeconds;
                    double nbStepsByEpoch = ((double)trainingDataset.Count) / miniBatchSizeForAllWorkers;
                    var msByStep = (1000 * secondsForEpoch) / nbStepsByEpoch;
                    LogInfo("Epoch " + epoch + "/" + Sample.num_epochs + " - " + Math.Round(secondsForEpoch, 0) + "s " + Math.Round(msByStep, 0) + "ms/step - lr: " + Math.Round(learningRateAtEpochStart, 8) + " - " + lossAndAccuracyMsg + " - " + ModelName);
                    LogDebug(MemoryInfo());
                    //if it is the last epoch, we'll save Layer KPI
                    if (epoch == Sample.num_epochs)
                    {
                        LogInfo(LayersKpi());
                    }

                    #region we save stats about the just finished epoch
                    var currentEpochData = new EpochData(epoch, learningRateAtEpochStart, lrMultiplicativeFactorFromReduceLrOnPlateau, secondsForEpoch, trainingMetrics, validationMetrics);
                    EpochData.Add(currentEpochData);
                    #endregion

                    var modelNameForEpoch = (epoch == Sample.num_epochs) ? ModelName : (ModelName + "_" + epoch);
                    #region we save the network in disk if necessary
                    bool isNewBestIterationToSave = IsNewBestIterationToSave();
                    if (ShouldSaveNetwork(learningRateComputer, savedNetworks, epoch) || isNewBestIterationToSave)
                    {
                        Debug.Assert(save != null);
                        if (isNewBestIterationToSave)
                        {
                            LogInfo("Saving predictions because of new best validation score");
                            Utils.TryDelete(savedNetworks.Values.SelectMany(x => x));
                            savedNetworks.Clear();
                        }
                        var savedFiles = save(true, isNewBestIterationToSave, trainingDataset, validationDatasetIfAny, modelNameForEpoch);
                        savedNetworks[DateTime.Now] = savedFiles;
                    }
                    #endregion

                    if (Sample.SaveNetworkStatsAfterEachEpoch)
                    {
                        var networkStatFileName = Path.Combine(WorkingDirectory, modelNameForEpoch + "_NetworkStats.txt");
                        LogInfo("Saving network '" + ModelName + "' stats in " + networkStatFileName);
                        File.WriteAllText(networkStatFileName, ContentStats());
                    }
                    if (ShouldStopTrainingBecauseOfEarlyStopping(EpochData, Sample.EarlyStoppingRounds, Sample.GetLoss()))
                    {
                        LogInfo("Stopping Training because of EarlyStopping");
                        break;
                    }
                }

                LogInfo("Training '"+ ModelName+"' for " + Sample.num_epochs + " epochs took: " + _spInternalFit.Elapsed.TotalSeconds + "s");
                if (!string.IsNullOrEmpty(ModelName))
                {
                    LogDebug("Network Name: "+ModelName);
                }
                _spInternalFit.Stop();
            }
            catch (Exception e)
            {
                LogError(e.ToString());
                throw;
            }


            return (null, null, null, null, null, null,
                ModelScore(Sample.GetLoss(), false), //trainLossIfAvailable,
                ModelScore(Sample.GetLoss(), true),  //validationLossIfAvailable,
                ModelScore(Sample.GetRankingEvaluationMetric(), false), //trainRankingMetricIfAvailable,
                ModelScore(Sample.GetRankingEvaluationMetric(), true)); //validationRankingMetricIfAvailable
        }

        private bool ShouldSaveNetwork(ILearningRateComputer learningRateComputer, Dictionary<DateTime, List<string>> savedNetworks, int epoch)
        {
            if (Sample.AutoSaveIntervalInMinutes < 0)
            {
                return false;
            }
            if (CurrentValidationScoreIsBelowMinimumRankingScoreToSaveModel())
            {
                return false;
            }
            return   //if we have finished training
                     (epoch == Sample.num_epochs && Sample.num_epochs >= 10)
                     //or if we should save the network every 'Config.AutoSaveIntervalInMinutes' minutes
                     || (savedNetworks.Count>0 && (DateTime.Now - savedNetworks.Keys.Max()).TotalMinutes > Sample.AutoSaveIntervalInMinutes)
                     || learningRateComputer.ShouldCreateSnapshotForEpoch(epoch)
                     || ShouldStopTrainingBecauseOfEarlyStopping(EpochData, Sample.EarlyStoppingRounds, Sample.GetLoss())
                     ;
        }

        /// <summary>
        /// check if we have just reached an epoch with a new best validation score that is high enough that we should save the network and the predictions
        /// </summary>
        /// <returns></returns>
        private bool CurrentValidationScoreIsBelowMinimumRankingScoreToSaveModel()
        {
            return CurrentRankingValidationScore() != null
                   && Sample.GetMinimumRankingScoreToSaveModel() != null
                   && Sample.GetMinimumRankingScoreToSaveModel().IsBetterThan(CurrentRankingValidationScore());
        }

        private IScore CurrentRankingValidationScore()
        {
            if (EpochData.Last().ValidationMetrics.TryGetValue(Sample.GetRankingEvaluationMetric(), out var rankingLastEpoch) && !double.IsNaN(rankingLastEpoch))
            {
                return new Score((float)rankingLastEpoch, Sample.GetRankingEvaluationMetric());
            }
            return null;
        }

        /// <summary>
        /// check if we have just reached an epoch with a new best validation score that is high enough that we should save the network and the predictions
        /// </summary>
        /// <returns></returns>
        private bool IsNewBestIterationToSave()
        {
            return CurrentRankingValidationScore() != null
                   && BestValidationRankingScore().epochBestScore == EpochData.Count
                   && Sample.GetMinimumRankingScoreToSaveModel() != null
                   && CurrentRankingValidationScore().IsBetterThan(Sample.GetMinimumRankingScoreToSaveModel());
        }

        /// <summary>
        /// return the epoch (between 1 and EpochData.Count) with the best validation ranking score and the associate score
        /// returns -1 if no validation ranking score is available
        /// </summary>
        /// <returns></returns>
        private (int epochBestScore, IScore bestScore) BestValidationRankingScore() => ModelBestScore(Sample.GetRankingEvaluationMetric(), true);

        private IScore ModelScore(EvaluationMetricEnum metric, bool useValidationDataset)
        {
            if (Sample.use_best_model)
            {
                return ModelBestScore(metric, useValidationDataset).bestScore;
            }
            for (int i = EpochData.Count-1; i >=0;--i)
            {
                var knownMetrics = useValidationDataset ? EpochData[i].ValidationMetrics : EpochData[i].TrainingMetrics;
                if (knownMetrics.TryGetValue(metric, out var score) && !double.IsNaN(score))
                {
                    return new Score((float)score, metric);
                }
            }
            return null;
        }

        /// <summary>
        /// return the epoch (between 1 and EpochData.Count) with the best score for metric 'metric'
        /// will use the validation Dataset score if useValidationDataset==true, and the training dataset score otherwise
        /// returns -1 if no score for metric 'metric' is available for the required dataset
        /// </summary>
        /// <param name="metric"></param>
        /// <param name="useValidationDataset">true for validation Dataset, false for training Dataset</param>
        /// <returns></returns>
        private (int epochBestScore, IScore bestScore) ModelBestScore(EvaluationMetricEnum metric, bool useValidationDataset)
        {
            int epochBestScore = -1;
            IScore bestScore = null;
            for (int i = 0; i < EpochData.Count; ++i)
            {
                var knownMetrics = useValidationDataset ? EpochData[i].ValidationMetrics : EpochData[i].TrainingMetrics;
                if (!knownMetrics.TryGetValue(metric, out var score) || double.IsNaN(score))
                {
                    continue;
                }
                var epochScore = new Score((float)score, metric);
                if (bestScore == null || epochScore.IsBetterThan(bestScore))
                {
                    bestScore = epochScore;
                    epochBestScore = i + 1;
                }
            }
            return (epochBestScore, bestScore);
        }

        /// <summary>
        /// if the validation loss has degraded for several consecutive epochs  (at least 'earlyStoppingRounds')
        /// then we stop the training
        /// </summary>
        /// <returns></returns>
        private static bool ShouldStopTrainingBecauseOfEarlyStopping(List<EpochData> epochData, int earlyStoppingRounds, EvaluationMetricEnum evaluationMetric)
        {
            if (earlyStoppingRounds <= 0)
            {
                return false;
            }
            // number of consecutive epochs where the validation loss has degraded.
            int nbConsecutiveEpochsWithDegradationOfValidationLoss = 0;
            for (int i = epochData.Count - 1; i >= 1; --i)
            {
                var currentValidationLoss = epochData[i].GetValidationLoss(evaluationMetric);
                var previousValidationLoss = epochData[i - 1].GetValidationLoss(evaluationMetric);
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

        private bool ShouldUseFullTestDataSetForLossAndAccuracy(ILearningRateComputer learningRateComputer, int epoch)
        {
            if (Sample.AlwaysUseFullTestDataSetForLossAndAccuracy || epoch == 1 || epoch == Sample.num_epochs)
            {
                return true;
            }
            return learningRateComputer.ShouldCreateSnapshotForEpoch(epoch);
        }

        #region compute Loss and Accuracy
        public List<KeyValuePair<EvaluationMetricEnum, double>> ComputeMetricsForValidationDataSet(int miniBatchSize, DataSet validationDataSet)
        {
            //We perform a mini batch gradient descent in Validation mode:
            //  there will be no shuffling/data augmentation.
            var (_,metrics) = MiniBatchGradientDescentForSingleEpoch(validationDataSet, miniBatchSize, returnPredictionsForFullDataset:false, computeMetricsForFullDataset:true);
            return metrics;
        }
        #endregion

        #region PyTorch support
        public override string ToPytorchModule(int batch_size)
        {
            var constructorLines = new List<string>(new []{"super().__init__()", "torch.manual_seed(0)", "np.random.seed(0)"});
            List<string> forwardLines = new();
            foreach (var layer in Layers)
            {
                layer.ToPytorchModule(constructorLines, forwardLines);
            }
            var last_output_variable = forwardLines[^1].Split(new[] { ' ', '=' })[0];
            forwardLines.Add("return "+ last_output_variable);

            var sb = new StringBuilder();
            sb.AppendLine("");
            sb.AppendLine("import torch");
            sb.AppendLine("import numpy as np");
            sb.AppendLine("import torch.nn.functional as F");
            sb.AppendLine("device = 'cuda'");

            sb.AppendLine(Sample.ToPytorchModule(this));

            if (Sample is not EfficientNetNetworkSample)
            {
                sb.AppendLine("");
                sb.AppendLine("class " + ModelName + "(torch.nn.Module):");
                sb.AppendLine("    def __init__(self):");
                foreach (var line in constructorLines)
                {
                    sb.AppendLine("        " + line);
                }

                sb.AppendLine("");
                sb.AppendLine("    def forward(self, x: torch.Tensor) -> torch.Tensor:");
                foreach (var line in forwardLines)
                {
                    sb.AppendLine("        " + line);
                }
                sb.AppendLine();
                sb.AppendLine("model = " + ModelName + "().to(device)");
            }

            sb.AppendLine("sample = dict()");
            sb.AppendLine("sample['num_epochs'] = " + Sample.num_epochs);
            sb.AppendLine("sample['batch_size'] = " + Sample.BatchSize);
            sb.AppendLine("# batch_size = " + Sample.BatchSize);
            sb.AppendLine("# epochs = " + Sample.num_epochs);
            sb.AppendLine("loss_criterion = " + Sample.PytorchLoss());
            sb.AppendLine("optimizer = "+Sample.PytorchOptimizer());
            if (Sample.LearningRateSchedulerType == NetworkSample.LearningRateSchedulerEnum.OneCycle)
            {
                sb.AppendLine($"scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr={Sample.InitialLearningRate}, steps_per_epoch=len(training_dataset)//batch_size, epochs=epochs, pct_start={(1-Sample.OneCycle_PercentInAnnealing)/2}, anneal_strategy='linear', div_factor={Sample.OneCycle_DividerForMinLearningRate}, three_phase=True )");
            }
            else
            {
                sb.AppendLine($"scheduler = None");
            }
            var shape_numpy_array_for_tests = "["+batch_size+", "+string.Join(", ", Layers[0].OutputShape(666).Skip(1))+"]";

            sb.AppendLine("# loss_before, loss_after = Train(model,");
            sb.AppendLine("Train(model,");
            sb.AppendLine("#    numpy_array_for_tests("+ shape_numpy_array_for_tests+"),");
            var shape_y_numpy_array_for_tests =batch_size+ ", " + string.Join(", ", Layers.Last().OutputShape(666).Skip(1));
            sb.AppendLine("#    y_numpy_array_for_tests("+ shape_y_numpy_array_for_tests+"),");
            sb.AppendLine("    training_dataset,");
            sb.AppendLine("    validation_dataset,");
            sb.AppendLine("    #device = device,");
            sb.AppendLine("    loss_criterion = loss_criterion");
            sb.AppendLine("    optimizer = optimizer,");
            sb.AppendLine("    scheduler = scheduler,");
            sb.AppendLine("    sample = sample,");
            sb.AppendLine("    #num_epochs = epochs,");
            sb.AppendLine("    #batch_size = batch_size");
            sb.AppendLine("    )");

            return sb.ToString();
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
        public override (DataFrame, string) PredictWithPath(DataSet dataset, bool removeAllTemporaryFilesAtEnd)
        {
            var cpuTensor = Predict(dataset, Sample.BatchSize);
            if (dataset.DatasetSample.TargetLabels.Length == cpuTensor.Shape[1])
            {
                return (DataFrame.New(cpuTensor, dataset.DatasetSample.TargetLabels), "");
            }
            return (DataFrame.New(cpuTensor), "");
        }
        public int TotalParams()
        {
            return Layers.SelectMany(l => l.Parameters).Select(t => t.Item1.Count).Sum();
        }
        private int[] YExpected_MiniBatch_Shape(int miniBatchSize)
        {
            var YPredicted_MiniBatch_Shape = this.YPredicted_MiniBatch_Shape(miniBatchSize);
            if (Utils.Product(YPredicted_MiniBatch_Shape) <= 0)
            {
                throw new ArgumentException($"invalid {nameof(YPredicted_MiniBatch_Shape)} shape: {Utils.ShapeToString(YPredicted_MiniBatch_Shape)}");
            }

            if (Sample.GetLoss() == EvaluationMetricEnum.SparseCategoricalCrossentropy)
            {
                // the Y Predicted shape is of shape (a, embeddingDim)
                // the Y Expected shape is of shape  (a,1)
                if (YPredicted_MiniBatch_Shape.Length == 2)
                {
                    YPredicted_MiniBatch_Shape[1] = 1;
                    return YPredicted_MiniBatch_Shape;
                }
                // the Y Predicted shape is of shape (a,b, ...,y,z, embeddingDim)
                // the Y Expected shape is of shape  (a,b, ...,y,z)
                return YPredicted_MiniBatch_Shape.Take(YPredicted_MiniBatch_Shape.Length - 1).ToArray();
            }
            return YPredicted_MiniBatch_Shape;
        }
        public int[] YPredicted_MiniBatch_Shape(int miniBatchSize)
        {
            return Layers.Last().OutputShape(miniBatchSize);
        }

        public CpuTensor<float> Predict(DataSet dataset, int miniBatchSizeForAllWorkers)
        {
            var (yPredicted,_) = MiniBatchGradientDescentForSingleEpoch(dataset, miniBatchSizeForAllWorkers, returnPredictionsForFullDataset: true, computeMetricsForFullDataset: false);
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
        /// <param name="computeMetricsForFullDataset"></param>
        /// <param name="returnPredictionsForFullDataset"></param>
        /// <returns>observed output associated with the input 'x'</returns>
        public (Tensor yPredictions, List<KeyValuePair<EvaluationMetricEnum, double>> metrics) MiniBatchGradientDescentForSingleEpoch(DataSet dataSet, int miniBatchSizeForAllWorkers, ILearningRateComputer learningRateComputerIfTraining = null, Action<Tensor, Tensor> CallBackAfterEachMiniBatch = null, bool returnPredictionsForFullDataset = true, bool computeMetricsForFullDataset = false)
        {
            Debug.Assert(IsMaster);
            Debug.Assert(miniBatchSizeForAllWorkers >= 1);
            dataSet.StartBackgroundThreadToLoadNextMiniBatchIfNeeded();

            if (_slaveNetworks.Any())
            {
                CompactParameters();
                CompactGradients();
            }

            var metricAccumulatorForSingleEpoch = computeMetricsForFullDataset
                ? new EvaluationMetricAccumulatorForSingleEpoch(MemoryPool, dataSet.Count, Sample)
                : null;

            //last time we display a progress on the screen for the current min batch descent
            var miniBatchGradientDescentStart = DateTime.Now;
            var lastStatsUpdate = miniBatchGradientDescentStart;
            bool isTraining = learningRateComputerIfTraining != null;

            //the mini batch size must be a multiple of the number of workers
            Debug.Assert(miniBatchSizeForAllWorkers< DegreeOfParallelism || miniBatchSizeForAllWorkers % DegreeOfParallelism == 0);
            int miniBatchSizeForEachWorker = Math.Max(1, miniBatchSizeForAllWorkers / DegreeOfParallelism);

            //the first epoch is #1
            int epoch = EpochData.Count + 1;
            var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputerIfTraining?.MultiplicativeFactorFromReduceLrOnPlateau(EpochData, Sample.GetLoss()) ?? 1.0;

            //dataSet.Count:
            // actual number of elements in the dataSet that we'll process
            //dataSetCountWithExtraBufferAtEnd:
            // Length of the tensors used to store the expected (_yExpectedForEpoch) & predicted values (_yPredictedForEpoch)
            // Those tensors contains an extra buffer of 'miniBatchSizeForAllWorkers-1' elements at the end
            // to make sure we can always split the dataSet in batch of exactly 'miniBatchSizeForAllWorkers' elements
            //int multiplier = (dataSet.Count + miniBatchSizeForAllWorkers - 1) / miniBatchSizeForAllWorkers;
            //int dataSetCountWithExtraBufferAtEnd = miniBatchSizeForAllWorkers* multiplier;
            int dataSetCountWithExtraBufferAtEnd = dataSet.Count + miniBatchSizeForAllWorkers - 1;


            if (returnPredictionsForFullDataset)
            {
                var YPredictedShapeAsMultipleOfMiniBatchSize = YPredicted_MiniBatch_Shape(dataSetCountWithExtraBufferAtEnd);
                MemoryPool.GetFloatTensor(ref _yPredictedForEpoch, YPredictedShapeAsMultipleOfMiniBatchSize);
            }
            else
            {
                MemoryPool.FreeFloatTensor(ref _yPredictedForEpoch);
                Debug.Assert(_yPredictedForEpoch == null);
            }

            //we create the shuffled list of inputs 
            var shuffledElementId = Enumerable.Range(0, dataSetCountWithExtraBufferAtEnd).ToArray();
            for (int i = dataSet.Count; i < shuffledElementId.Length; ++i)
            {
                shuffledElementId[i] = i%dataSet.Count;
            }

            if (Sample.ShuffleDatasetBeforeEachEpoch && isTraining)
            {
                if (Sample.ShuffleDatasetBeforeEachEpochBlockSize <= 1)
                {
                    Utils.Shuffle(shuffledElementId, Rand);
                }
                else
                {
                    Utils.Shuffle(shuffledElementId, Rand, Sample.ShuffleDatasetBeforeEachEpochBlockSize);
                }
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
            var yExpected_miniBatch_cpu_allWorkers = new CpuTensor<float>(YExpected_MiniBatch_Shape(miniBatchSizeForAllWorkers), null);
            var shuffledElementIdMemory = new Memory<int>(shuffledElementId);

            var yExpected_miniBatch_allWorkers = 
                MemoryPool.GetFloatTensor(YExpected_MiniBatch_Shape(miniBatchSizeForAllWorkers));

            var yPredicted_miniBatch_allWorkers = _yPredictedForEpoch==null
                ? MemoryPool.GetFloatTensor(YPredicted_MiniBatch_Shape(miniBatchSizeForAllWorkers))
                : null;

            for (int firstIndexInShuffledElementId = 0; firstIndexInShuffledElementId < dataSet.Count; )
            {
                if (_yPredictedForEpoch != null)
                {
                    yPredicted_miniBatch_allWorkers = _yPredictedForEpoch.RowSlice(firstIndexInShuffledElementId, miniBatchSizeForAllWorkers);
                }
                Debug.Assert(yPredicted_miniBatch_allWorkers != null);
                
                //LogInfo("Processing epoch " + epoch + " for elements [" + firstIndexInShuffledElementId + ":]");

                //we initialize miniBatch input (xMiniBatch) and expected output (yExpectedMiniBatchCpu)
                StartTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                bool withDataAugmentation = Sample.UseDataAugmentation && isTraining;
                int actualNumberOfLoadedItems = dataSet.LoadMiniBatch(withDataAugmentation, isTraining, shuffledElementId, firstIndexInShuffledElementId, Sample, all_x_miniBatch_cpu_allWorkers, yExpected_miniBatch_cpu_allWorkers);
                #if DEBUG
                foreach (var t in all_x_miniBatch_cpu_allWorkers)
                {
                    var span = t.SpanContent;
                    for (var index = 0; index < span.Length; index++)
                    {
                        if (float.IsNaN(span[index]))  { throw new Exception($"NaN in input X Tensor at index {index}"); }
                    }
                }
                #endif
                StopTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                //we copy yExpected_miniBatch_cpu_allWorkers from CPU to appropriate target (CPU or GPU)
                yExpected_miniBatch_cpu_allWorkers.CopyTo(yExpected_miniBatch_allWorkers);

                //we launch the forward & backward computation on all slave networks
                var usedSlaves = new List<Network>();
                for (var slaveIndex = 0; slaveIndex < _slaveNetworks.Count; slaveIndex++)
                {
                    var slave = _slaveNetworks[slaveIndex];
                    var firstRowIndexForSlave = (1 + slaveIndex) * miniBatchSizeForEachWorker;
                    List<Tensor> x_miniBatch_cpu_slave = Tensor.RowSlice(all_x_miniBatch_cpu_allWorkers, firstRowIndexForSlave, miniBatchSizeForEachWorker);
                    var yExpected_miniBatch_cpu_slave = yExpected_miniBatch_cpu_allWorkers.RowSlice(firstRowIndexForSlave, miniBatchSizeForEachWorker);
                    var yPredicted_miniBatch_slave = yPredicted_miniBatch_allWorkers.RowSlice( firstRowIndexForSlave, miniBatchSizeForEachWorker);
                    slave._slaveParamForMiniBatchGradientDescent = Tuple.Create(x_miniBatch_cpu_slave, yExpected_miniBatch_cpu_slave, yPredicted_miniBatch_slave, isTraining);
                    slave._slaveStatus = SLAVE_NETWORK_STATUS.PERFORM_FORWARD_AND_BACKWARD_PROPAGATION;
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
                var yPredicted_miniBatch_master = yPredicted_miniBatch_allWorkers.RowSlice(0, miniBatchSizeForEachWorker);
                var yExpected_miniBatch_master = yExpected_miniBatch_allWorkers.RowSlice(0, miniBatchSizeForEachWorker);
                PropagationManager.Forward(all_x_miniBatch, yPredicted_miniBatch_master, isTraining);
                if (isTraining)
                {
                    PropagationManager.Backward(yExpected_miniBatch_master, yPredicted_miniBatch_master, Sample);
                }

                // we ensure that all slaves have finished
                if (_slaveNetworks.Any())
                { 
                    //we wait for all slave to finish the forward & backward propagation pass
                    StartTimer("WaitForSlave_Forward", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                    WaitForAllSlavesInStatus(SLAVE_NETWORK_STATUS.IDLE);
                    StopTimer("WaitForSlave_Forward", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                }

                // we update the weights
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

                //We update the Evaluation Metrics if needed
                metricAccumulatorForSingleEpoch?.UpdateMetrics(yExpected_miniBatch_allWorkers, yPredicted_miniBatch_allWorkers);

                CallBackAfterEachMiniBatch?.Invoke(yExpected_miniBatch_master, yPredicted_miniBatch_master);

                if ((DateTime.Now-lastStatsUpdate).TotalSeconds> 10*60)
                {
                    var lastIndexInShuffledElementId = firstIndexInShuffledElementId + miniBatchSizeForAllWorkers - 1;
                    var percentageDoneInEpoch = ((double) lastIndexInShuffledElementId) / dataSetCountWithExtraBufferAtEnd;
                    var secondsSinceStartOfEpoch = (DateTime.Now - miniBatchGradientDescentStart).TotalSeconds;
                    var expectedSecondsToPerformEntireEpoch = secondsSinceStartOfEpoch / percentageDoneInEpoch;
                    LogInfo((isTraining ? ("Epoch " + epoch) : "Inference") + " in progress: " + Math.Round(100.0 * percentageDoneInEpoch, 1) + "% performed (" + Math.Round(secondsSinceStartOfEpoch, 0) + "s/" + Math.Round(expectedSecondsToPerformEntireEpoch, 0) + "s)");
                    LogDebug(MemoryInfo());
                    lastStatsUpdate = DateTime.Now;
                }
                firstIndexInShuffledElementId += actualNumberOfLoadedItems;
            }
            dataSet.StopBackgroundThreadToLoadNextMiniBatchIfNeeded();

            all_x_miniBatch_cpu_allWorkers.ForEach(t => t.Dispose());
            all_x_miniBatch_cpu_allWorkers.Clear();
            yExpected_miniBatch_cpu_allWorkers.Dispose();
            MemoryPool.FreeFloatTensor(ref yExpected_miniBatch_allWorkers);
            if (_yPredictedForEpoch == null)
            {
                MemoryPool.FreeFloatTensor(ref yPredicted_miniBatch_allWorkers);
            }

            _yPredictedForEpoch?.ReshapeInPlace(YPredicted_MiniBatch_Shape(dataSet.Count));

            var metrics = metricAccumulatorForSingleEpoch?.Metrics();
            metricAccumulatorForSingleEpoch?.Dispose();
            return (_yPredictedForEpoch, metrics);
        }
        public int LastLayerIndex => Layers.Last().LayerIndex;
        public int NbLayerOfType(Type layerType)
        {
            return Layers.Count(l => l.GetType() == layerType);
        }
        public bool UseGPU => Sample.MustUseGPU;
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
        public string Summary()
        {
            return Layers.Any(x => x.PreviousLayers.Count >= 2) ? SummaryWithConnectedTo() : SummaryWithoutConnectedTo();
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
        private string SummaryWithConnectedTo()
        {
            const int firstColumnWidth = 32;
            const int secondColumnWidth = 21;
            const int thirdColumnWidth = 12;
            const int forthColumnWidth = 33;
            var line0 = new string('_', firstColumnWidth + secondColumnWidth + thirdColumnWidth + forthColumnWidth);
            var line1 = new string('=', line0.Length);
            string result = "";
            if (!string.IsNullOrEmpty(ModelName))
            {
                result += "Network Name: " + ModelName + Environment.NewLine;
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
            result += "Total params: " + TotalParams().ToString("N0", CultureInfo.InvariantCulture)+Environment.NewLine;
            result += "Trainable params: " + (TotalParams()- NonTrainableParams).ToString("N0", CultureInfo.InvariantCulture) + Environment.NewLine;
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
            if (!string.IsNullOrEmpty(ModelName))
            {
                result += "Network Name: " + ModelName + Environment.NewLine;
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
            result += "Total params: " + TotalParams().ToString("N0", CultureInfo.InvariantCulture) + Environment.NewLine;
            result += "Trainable params: " + (TotalParams() - NonTrainableParams).ToString("N0", CultureInfo.InvariantCulture) + Environment.NewLine;
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
        private Tensor Predict(List<Tensor> allX, bool isTraining)
        {
            var batchSize = allX[0].Shape[0];
            var yPredicted = MemoryPool.GetFloatTensor(YPredicted_MiniBatch_Shape(batchSize));
            for (int i = 0; i < allX.Count; ++i)
            {
                allX[i] = ReformatToCorrectDevice_GPU_or_CPU(allX[i]);
            }
            PropagationManager.Forward(allX, yPredicted, isTraining);
            return yPredicted;
        }

    }
}

