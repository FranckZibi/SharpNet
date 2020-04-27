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
        private readonly List<EpochData> _epochData = new List<EpochData>();
        private readonly IDictionary<string,Stopwatch> _updateWeightsTime = new Dictionary<string, Stopwatch>();
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
        public IDictionary<string, Stopwatch> ForwardPropagationTrainingTime { get; } = new Dictionary<string, Stopwatch>();
        public IDictionary<string, Stopwatch> ForwardPropagationInferenceTime { get; } = new Dictionary<string, Stopwatch>();
        public IDictionary<string, Stopwatch> BackwardPropagationTime { get; } = new Dictionary<string, Stopwatch>();
        public TensorMemoryPool MemoryPool { get; }
        public GPUWrapper GpuWrapper { get; }
        private int GpuDeviceId { get; }
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
            GpuDeviceId = resourceIds[0];
            GpuWrapper = UseGPU ? GPUWrapper.FromDeviceId(GpuDeviceId) : null;
            _swComputeLoss = new Stopwatch();
            _swComputeAccuracy = new Stopwatch();
            CreateLogDirectoryIfNeeded();
            MemoryPool = new TensorMemoryPool(GpuWrapper, false);
            PropagationManager = new PropagationManager(Layers, MemoryPool, ForwardPropagationTrainingTime, ForwardPropagationInferenceTime, BackwardPropagationTime, _updateWeightsTime);

            if (IsMaster)
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


        #region Multi GPU Support

        private int DegreeOfParallelism
        {
            get
            {
                if (IsMaster)
                {
                    return 1 + _slaveNetworks.Count;
                }
                return _masterNetworkIfAny.DegreeOfParallelism;
            }
        }
        public bool IsMaster => _masterNetworkIfAny == null;

        #region Master Network data
        private Tensor _x_miniBatch;
        /// <summary>
        /// list of all slave networks
        /// empty list of:
        ///     the 'this' network is a slave network
        ///     or
        ///     the 'this' network is a master network  but we are not using multi GPU computation
        ///     (all computation is performed on the master network)
        /// </summary>
        private readonly List<Network> _slaveNetworks = new List<Network>();
        private Tensor _yExpectedForEpoch;
        private Tensor _yPredictedForEpoch;
        /// <summary>
        /// number of resources (CPU or GPU) that will perform the computation
        /// </summary>
        private void WaitForAllSlavesInIdleMode()
        {
            while (_slaveNetworks.Any(s => s._slaveStatus != SLAVE_NETWORK_STATUS.IDLE))
            {
                Thread.Sleep(1);
            }
        }
        private void SetStatusForAllSlaves(SLAVE_NETWORK_STATUS newStatus)
        {
            _slaveNetworks.ForEach(s => s._slaveStatus = newStatus);
        }
        #endregion

        #region Slave Networks data
        private enum SLAVE_NETWORK_STATUS
        {
            IDLE,
            PREPARE_MINIBATCH_GRADIENT_DESCENT,
            PERFORM_FORWARD_AND_BACKWARD_PROPAGATION,
            STOP_THREAD
        }
        private SLAVE_NETWORK_STATUS _slaveStatus = SLAVE_NETWORK_STATUS.IDLE;
        /// <summary>
        /// if the current network is a slave network doing computation for its master network:
        ///     the reference of the master network
        /// if current network is a master network:
        ///     null
        /// </summary>
        private readonly Network _masterNetworkIfAny;
        //Tensor x_miniBatch_cpu_slave, Tensor yExpected_miniBatch_cpu_slave, Tensor yPredicted_miniBatch_master, bool isTraining
        private Tuple<Tensor, Tensor, Tensor, bool> _slaveParamForMiniBatchGradientDescent;



        private static void SlaveThread(Network master, int slaveDeviceId)
        {
            var slave = new Network(master.Config.Clone(), new List<int> { slaveDeviceId }, master);
            slave.Config.Logger = slave.Config.Logger.CloneWithNewFileSuffix("_slave" + slaveDeviceId);
            slave.Description = master.Description + "_slave";
            lock (master._slaveNetworks)
            {
                master._slaveNetworks.Add(slave);
            }
            for (;;)
            {
                switch (slave._slaveStatus)
                {
                    case SLAVE_NETWORK_STATUS.PREPARE_MINIBATCH_GRADIENT_DESCENT:
                        if (slave.Layers.Count == 0)
                        {
                            foreach (var l in master.Layers)
                            {
                                slave.Layers.Add(l.CloneForSlaveNetwork(slave));
                            }
                        }
                        slave._slaveStatus = SLAVE_NETWORK_STATUS.IDLE;
                        break;
                    case SLAVE_NETWORK_STATUS.PERFORM_FORWARD_AND_BACKWARD_PROPAGATION:
                        var param = slave._slaveParamForMiniBatchGradientDescent;
                        if (param == null)
                        {
                            var errorMsg = "null parameters for " + slave._slaveStatus;
                            slave.Info(errorMsg);
                            throw new ArgumentException(errorMsg);
                        }
                        slave.MiniBatchGradientDescentForSlave(param.Item1, param.Item2, param.Item3, param.Item4);
                        slave._slaveParamForMiniBatchGradientDescent = null;
                        slave._slaveStatus = SLAVE_NETWORK_STATUS.IDLE;
                        break;
                    case SLAVE_NETWORK_STATUS.STOP_THREAD:
                        slave.Dispose();
                        return;
                    default:
                    //case SLAVE_NETWORK_STATUS.IDLE:
                        Thread.Sleep(1);
                        break;
                }
            }
        }
        /// <summary>
        /// Clone a tensor 't' coming from the master network to a tensor for the current (slave) network
        /// </summary>
        /// <param name="tFromMasterNetwork">the tensor from the master network to clone</param>
        /// <returns></returns>
        public Tensor CloneFromMasterNetwork(Tensor tFromMasterNetwork)
        {
            Debug.Assert(!IsMaster);
            if (tFromMasterNetwork == null)
            {
                return null;
            }
            var result = MemoryPool.GetNotInitializedFloatTensor((int[])tFromMasterNetwork.Shape.Clone(), tFromMasterNetwork.Description);
            tFromMasterNetwork.CopyTo(result);
            return result;
        }

        private void MiniBatchGradientDescentForSlave(Tensor x_miniBatch_cpu_slave, Tensor yExpected_miniBatch_cpu_slave, Tensor yPredicted_miniBatch_master, bool isTraining)
        {
            Debug.Assert(_yPredictedForEpoch == null);
            Debug.Assert(_yExpectedForEpoch == null);
            Debug.Assert(yExpected_miniBatch_cpu_slave.SameShape(yPredicted_miniBatch_master));
            Debug.Assert(x_miniBatch_cpu_slave.Shape[0] == yExpected_miniBatch_cpu_slave.Shape[0]);
            Debug.Assert(!yExpected_miniBatch_cpu_slave.UseGPU);
            Debug.Assert(!yPredicted_miniBatch_master.UseGPU);
            Debug.Assert(yPredicted_miniBatch_master.UseGPU == UseGPU);
            
            //We load the weights from the master network
            StartTimer("CopyWeights", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
            var parameters_master = _masterNetworkIfAny.Parameters;
            var parameters_slave = Parameters;
            Debug.Assert(parameters_master.Count == parameters_slave.Count);
            for (int i = 0; i < parameters_master.Count; ++i)
            {
                parameters_master[i].CopyTo(parameters_slave[i]);
            }
            StopTimer("CopyWeights", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);

            //we initialize '_xMiniBatch' & '_yExpected_miniBatch_slave'
            MemoryPool.GetNotInitializedFloatTensor(ref _x_miniBatch, x_miniBatch_cpu_slave.Shape, nameof(_x_miniBatch));
            x_miniBatch_cpu_slave.CopyTo(_x_miniBatch);
            MemoryPool.GetNotInitializedFloatTensor(ref _yExpected_miniBatch_slave, yExpected_miniBatch_cpu_slave.Shape, nameof(_yExpected_miniBatch_slave));
            yExpected_miniBatch_cpu_slave.CopyTo(_yExpected_miniBatch_slave);
            MemoryPool.GetNotInitializedFloatTensor(ref _yPredicted_miniBatch_slave, _yExpected_miniBatch_slave.Shape);
            PropagationManager.Forward(_x_miniBatch, _yPredicted_miniBatch_slave, isTraining);
            if (isTraining)
            {
                PropagationManager.Backward(_yExpected_miniBatch_slave, _yPredicted_miniBatch_slave);
            }

            //copy miniBatch prediction (computed in slave network) to master network
            _yPredicted_miniBatch_slave.CopyTo(yPredicted_miniBatch_master);
        }

        private Tensor _yExpected_miniBatch_slave;
        private Tensor _yPredicted_miniBatch_slave;

        #endregion




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

        /// <summary>
        /// Clone the current network
        /// </summary>
        /// <param name="slaveDeviceId"></param>
        /// <returns></returns>
        public string DeviceName() { return GpuWrapper?.DeviceName(); }
        public void Dispose()
        {
            if (IsMaster)
            { 
                SetStatusForAllSlaves(SLAVE_NETWORK_STATUS.STOP_THREAD);
                Thread.Sleep(100);
                _slaveNetworks.Clear();
            }
            LogDebug("Before clearing memory: " + MemoryInfo());
            GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
            GC.Collect();
            Layers.ForEach(l => l?.Dispose());
            Layers.Clear();
            PropagationManager.Dispose();
            _epochData.Clear();

            MemoryPool.FreeMemory(ref _bufferComputeAccuracy);
            MemoryPool.FreeMemory(ref _bufferComputeLoss);
            MemoryPool.FreeMemory(ref _yPredictedForEpoch);
            MemoryPool.FreeMemory(ref _yExpectedForEpoch);
            MemoryPool.FreeMemory(ref _x_miniBatch);
            MemoryPool.FreeMemory(ref _bufferAddGradientFromSlaveNetwork);
            MemoryPool.FreeMemory(ref _yExpected_miniBatch_slave);
            MemoryPool.FreeMemory(ref _yPredicted_miniBatch_slave);


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
            equals &= Utils.Equals(Description, other.Description, id + nameof(Description), ref errors);
            equals &= Config.Equals(other.Config, epsilon, id, ref errors);
            equals &= Utils.Equals(other.GpuDeviceId, GpuDeviceId, id, ref errors);
            equals &= Utils.Equals(Layers.Count, other.Layers.Count, id + "Layers.Count", ref errors);
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
        public int TotalParams => Parameters.Select(t=> t.Count).Sum();

        /// <summary>
        /// all parameters of the network (both trainable parameters and not trainable parameters)
        /// </summary>
        private List<Tensor> Parameters => Layers.SelectMany(x => x.Parameters).ToList();
        
        /// <summary>
        /// gradients of all trainable parameters of the network
        /// (there are no gradients for non trainable parameters)
        /// </summary>
        private List<Tensor> ParameterGradients => Layers.SelectMany(x => x.ParameterGradients).ToList();

     
        #region serialization
        // ReSharper disable once UnusedMember.Global
        public static Network ValueOf(string path, int?overrideGpuDeviceId = null)
        {
            var allLines = File.ReadAllLines(path);
            var dicoFirstLine = Serializer.Deserialize(allLines[0], null);
            var config = NetworkConfig.ValueOf(dicoFirstLine);
            var gpuDeviceId = overrideGpuDeviceId ?? (int)dicoFirstLine[nameof(GpuDeviceId)];
            var network = new Network(config, new List<int>{gpuDeviceId});
            var epochsData = (EpochData[])dicoFirstLine[nameof(_epochData)];
            network._epochData.AddRange(epochsData);
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
            var maxMiniBatchSizeForAllWorkers = DegreeOfParallelism*MaxMiniBatchSizeForEachWorker(trainingDataSet.XMiniBatch_Shape(1), false);
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
                MemoryPool.GetNotInitializedFloatTensor(ref _bufferComputeLoss, new[] { yExpectedMiniBatch.Shape[0] }, nameof(_bufferComputeLoss));
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
                
                Info(ToString());
                var maxMiniBatchSize = MaxMiniBatchSizeForEachWorker(trainingDataSetCpu.XMiniBatch_Shape(1), true);
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
                    var yPredicted = MiniBatchGradientDescentForSingleEpoch(trainingDataSetCpu, miniBatchSize, learningRateComputer, null);
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


            //last time we display a progress on the screen for the current min batch descent
            var miniBatchGradientDescentStart = DateTime.Now;
            var lastStatsUpdate = miniBatchGradientDescentStart;
            bool isTraining = learningRateComputerIfTraining != null;
            //total number of elements to process: they will be processed by block of 'miniBatchSize' elements
            var totalElementCount = dataSet.Count;
            if (miniBatchSizeForAllWorkers <= 0)
            {
                miniBatchSizeForAllWorkers = DegreeOfParallelism*MaxMiniBatchSizeForEachWorker(dataSet.XMiniBatch_Shape(1), isTraining);
            }

            //the mini batch size must be a multiple of the number of workers
            Debug.Assert(miniBatchSizeForAllWorkers< DegreeOfParallelism || miniBatchSizeForAllWorkers % DegreeOfParallelism == 0);
            int miniBatchSizeForEachWorker = Math.Max(1, miniBatchSizeForAllWorkers / DegreeOfParallelism);


            //the first epoch is #1
            int epoch = _epochData.Count + 1;
            var lrMultiplicativeFactorFromReduceLrOnPlateau = learningRateComputerIfTraining?.MultiplicativeFactorFromReduceLrOnPlateau(_epochData) ?? 1.0;
            MemoryPool.GetNotInitializedFloatTensor(ref _yPredictedForEpoch, dataSet.Y_Shape);
            MemoryPool.GetNotInitializedFloatTensor(ref _yExpectedForEpoch, dataSet.Y_Shape);

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

            StartTimer("WaitForSlave>Prepare", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
            SetStatusForAllSlaves(SLAVE_NETWORK_STATUS.PREPARE_MINIBATCH_GRADIENT_DESCENT);
            WaitForAllSlavesInIdleMode();
            StopTimer("WaitForSlave>Prepare", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);

   


            for (int firstIndexInShuffledElementId_master= 0; firstIndexInShuffledElementId_master< totalElementCount; firstIndexInShuffledElementId_master+= miniBatchSizeForAllWorkers)
            {
                var currentMiniBatchSize_allWorkers = Math.Min(totalElementCount- firstIndexInShuffledElementId_master, miniBatchSizeForAllWorkers);
                var currentMiniBatchSize_master = Math.Min(miniBatchSizeForEachWorker, currentMiniBatchSize_allWorkers);
                Debug.Assert(currentMiniBatchSize_master>=1);


                //we initialize miniBatch input (xMiniBatch) and expected output (yExpectedMiniBatchCpu)

                StartTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                var x_miniBatch_cpu_allWorkers = new CpuTensor<float>(dataSet.XMiniBatch_Shape(currentMiniBatchSize_allWorkers), null, "x_miniBatch_cpu_allWorkers");
                var yExpected_miniBatch_cpu_allWorkers = new CpuTensor<float>(dataSet.YMiniBatch_Shape(currentMiniBatchSize_allWorkers), null, "yExpected_miniBatch_cpu_allWorkers");
                bool withDataAugmentation = Config.DataAugmentation.UseDataAugmentation && (epoch >= 2) && isTraining;
                dataSet.LoadMiniBatch(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementId_master, Config.DataAugmentation, x_miniBatch_cpu_allWorkers, yExpected_miniBatch_cpu_allWorkers);
                StopTimer("LoadInput", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                //we copy yExpected_miniBatch_cpu_allWorkers from CPU to appropriate target (CPU or GPU)
                var yExpectedForMiniBatch_allWorkers = _yExpectedForEpoch.Slice(firstIndexInShuffledElementId_master, currentMiniBatchSize_allWorkers);
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
                    var x_miniBatch_cpu_slave = x_miniBatch_cpu_allWorkers.Slice(firstIndexInShuffledElement_slave - firstIndexInShuffledElementId_master, currentMiniBatchSize_slave);
                    var yExpected_miniBatch_cpu_slave = yExpected_miniBatch_cpu_allWorkers.Slice(firstIndexInShuffledElement_slave - firstIndexInShuffledElementId_master, currentMiniBatchSize_slave);
                    var yPredicted_miniBatch_slave = _yPredictedForEpoch.Slice(firstIndexInShuffledElement_slave, currentMiniBatchSize_slave);
                    slave._slaveParamForMiniBatchGradientDescent = Tuple.Create(x_miniBatch_cpu_slave, yExpected_miniBatch_cpu_slave, yPredicted_miniBatch_slave, isTraining);
                    slave._slaveStatus = SLAVE_NETWORK_STATUS.PERFORM_FORWARD_AND_BACKWARD_PROPAGATION;
                    firstIndexInShuffledElement_slave += currentMiniBatchSize_slave;
                    usedSlaves.Add(slave);
                }

                //we launch the forward & backward propagation on the master network
                var x_miniBatch_cpu_master = x_miniBatch_cpu_allWorkers.Slice(0, currentMiniBatchSize_master);
                MemoryPool.GetNotInitializedFloatTensor(ref _x_miniBatch, x_miniBatch_cpu_master.Shape, nameof(_x_miniBatch));
                x_miniBatch_cpu_master.CopyTo(_x_miniBatch);
                var yPredicted_miniBatch_master = _yPredictedForEpoch.Slice(firstIndexInShuffledElementId_master, currentMiniBatchSize_master);
                var yExpected_miniBatch_master = _yExpectedForEpoch.Slice(firstIndexInShuffledElementId_master, currentMiniBatchSize_master);
                PropagationManager.Forward(_x_miniBatch, yPredicted_miniBatch_master, isTraining);
                if (isTraining)
                {
                    PropagationManager.Backward(yExpected_miniBatch_master, yPredicted_miniBatch_master);
                }

                //we wait for all slave to finish the forward & backward propagation pass
                StartTimer("WaitForSlave>Forward", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);
                WaitForAllSlavesInIdleMode();
                StopTimer("WaitForSlave>Forward", isTraining ? ForwardPropagationTrainingTime : ForwardPropagationInferenceTime);

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
            return _yPredictedForEpoch;
        }


        private Tensor _bufferAddGradientFromSlaveNetwork;

        private void AddGradientFromSlaveNetwork(Network slave)
        {
            Debug.Assert(IsMaster);
            Debug.Assert(!slave.IsMaster);
            var slaveGradients = slave.ParameterGradients;
            var masterGradients = ParameterGradients;
            Debug.Assert(slaveGradients.Count == masterGradients.Count);
            for (int i = 0; i < masterGradients.Count; ++i)
            {
                var masterTensor = masterGradients[i];
                var slaveTensor = slaveGradients[i];
                Debug.Assert(masterTensor.SameShape(slaveTensor));
                MemoryPool.GetNotInitializedFloatTensor(ref _bufferAddGradientFromSlaveNetwork, masterTensor.Shape, nameof(_bufferAddGradientFromSlaveNetwork));
                slaveTensor.CopyTo(_bufferAddGradientFromSlaveNetwork); //Device to other Device copy (not in the same GPU)
                masterTensor.Update_Adding_Alpha_X(1, _bufferAddGradientFromSlaveNetwork);
            }
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
            var x = mockMemoryPooling.GetNotInitializedFloatTensor(xShape, "x");
            var yPredicted = mockMemoryPooling.GetNotInitializedFloatTensor(Layers.Last().OutputShape(x.Shape[0]), "yPredicted");
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
                    l.ResetWeights();
                }
            }
        }
        private int MaxMiniBatchSizeForEachWorker(int[] xShape, bool isTraining)
        {
            var bytesByBatchSizeForwardAndBackward = BytesByBatchSizeForwardAndBackward(xShape, isTraining);
            var freeMemoryInBytes = UseGPU ? (ulong)GpuWrapper.AvailableGpuMemoryInBytes() : Utils.AvailableRamMemoryInBytes();
            var maxMiniBatchSize = MaxMiniBatchSizeForEachWorker(bytesByBatchSizeForwardAndBackward, freeMemoryInBytes);
            LogDebug("Max MiniBatchSize=" + maxMiniBatchSize + " (free memory=" + Utils.MemoryBytesToString(freeMemoryInBytes) + ")");
            return maxMiniBatchSize;
        }
        //TODO add tests
        private static int MaxMiniBatchSizeForEachWorker(ulong bytesByBatchSize, ulong freeMemoryInBytes)
        {
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
