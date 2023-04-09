using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Models;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    public abstract class Layer : IDisposable
    {
        #region fields
        public int LayerIndex { get; }

        public string LayerName
        {
            get
            {
                if (string.IsNullOrEmpty(_layerName))
                {
                    _layerName = ComputeLayerName();
                }
                return _layerName;
            }
        }
        public List<int> NextLayerIndexes { get; } = new List<int>();
        public List<int> PreviousLayerIndexes { get; } = new List<int>();
        /// <summary>
        /// true if the layer weights should be updated during training
        /// false if it is a frozen layer
        /// </summary>
        public bool Trainable { get; set; } = true;
        protected readonly Network Network;
        protected bool _isDisposed;
        public int[] LazyOutputShape { private get; set; }
        private string _layerName;
        #endregion

        #region constructors
        protected Layer(Network network, int[] previousLayerIndexes, string layerName)
        {
            Network = network;
            LayerIndex = network.Layers.Count;
            // ReSharper disable once VirtualMemberCallInConstructor
            _layerName = layerName;
            foreach (var previousLayerIndex in previousLayerIndexes)
            {
                Debug.Assert(previousLayerIndex>=0);
                Debug.Assert(previousLayerIndex< LayerIndex);
                PreviousLayerIndexes.Add(previousLayerIndex);
                Layers[previousLayerIndex].NextLayerIndexes.Add(LayerIndex);
            }
            network.OnLayerAddOrRemove();
        }
        protected Layer(Network network, string layerName) : this(network, new[]{network.Layers.Count - 1}, layerName)
        {
        }
        #endregion

        #region forward and backward propagation
        /// <summary>
        ///  At this stage, we already know 'x', we want to compute 'y'
        /// </summary>
        /// <param name="allX"></param>
        /// <param name="y"></param>
        /// <param name="isTraining">true if we are currently training the network
        ///     false if we are just using it to make a prediction </param>
        public abstract void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining);
        /// <summary>
        ///  At this stage, we already know 'x'+'y'+'dy', we want to compute 'dx' + 'dw' by backward propagation
        /// </summary>
        /// <param name="allX">[in] all layer inputs</param>
        /// <param name="y">[in] the already computed output</param>
        /// <param name="dy">[in] the already computed output gradient</param>
        /// <param name="dx">[out] input gradient (dx) to compute from the output gradient (dy)</param>
        public abstract void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx);

        /// <summary>
        /// true if the output feature map 'y' is needed to compute the backward propagation of current layer
        /// </summary>
        public virtual bool OutputNeededForBackwardPropagation => true;
        /// <summary>
        /// true if the input feature map 'x' is needed to compute the backward propagation of current layer
        /// </summary>
        public virtual bool InputNeededForBackwardPropagation => true;


        /// <summary>
        /// update all weights of the layer thanks to the weight (& bias)  gradients computed in the backward propagation step
        /// only used in master network
        /// </summary>
        /// <param name="batchSize"></param>
        /// <param name="learningRate"></param>
        /// <param name="maxLearningRate"></param>
        public virtual void UpdateWeights(int batchSize, double learningRate, double maxLearningRate)
        {
            Debug.Assert(Network.IsMaster);
            if (Weights == null)
            {
                return;
            }
            Debug.Assert(Weights == null || Weights.SameShape(WeightGradients));
            Debug.Assert(Bias == null || Bias.SameShape(Bias));
            if (Trainable)
            {
                Optimizer?.UpdateWeights(learningRate, maxLearningRate, batchSize, Weights, WeightGradients, Bias, BiasGradients);
            }
        }
        #endregion

        #region parameters and gradients
        /// <summary>
        /// the weight if any
        /// </summary>
        public virtual Tensor Weights => null;
        /// <summary>
        /// the weight gradient if any
        /// </summary>
        public virtual Tensor WeightGradients => null;
        /// <summary>
        /// the bias if any
        /// </summary>
        public virtual Tensor Bias => null;
        /// <summary>
        /// the bias gradient if any
        /// </summary>
        public virtual Tensor BiasGradients => null;
        protected virtual Optimizer Optimizer => null;

        public int TotalParams => Parameters.Select(t => t.Item1.Count).Sum();
        public virtual int NonTrainableParams => 0;
        public virtual List<Tuple<Tensor, string>> Parameters => new();

        public virtual void ReplaceParameters(List<Tensor> newParameters)
        {
            Debug.Assert(!HasParameters);
        }
        public virtual int DisableBias()
        {
            return PreviousLayers.Select(l => l.DisableBias()).Sum();
        }
        public virtual void ReplaceGradients(List<Tensor> newGradients)
        {
            Debug.Assert(!HasParameters);
        }
        public virtual List<Tensor> ParameterGradients
        {
            get
            {
                var result = new List<Tensor> { WeightGradients, BiasGradients };
                result.RemoveAll(t => t == null);
                return result;
            }
        }
        public virtual void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            Debug.Assert(Network.IsMaster);
            Debug.Assert(!HasParameters);
        }


        #region *.h5 file (HDF) management
        /// <summary>
        /// Initialize layer weights & bias from datasets objects extracted from a *.h5 (HDF) file
        /// </summary>
        /// <param name="h5FileDataset">all datasets objects in the *.h5 file</param>
        /// <param name="originFramework">the ML Framework from where the *.h5 file comes from</param>
        // ReSharper disable once UnusedParameter.Global
        public virtual void LoadParameters(IDictionary<string, Tensor> h5FileDataset, NetworkSample.CompatibilityModeEnum originFramework)
        {
            foreach (var layerParameters in Parameters)
            {
                var parameterId = layerParameters.Item2;
                if (h5FileDataset.ContainsKey(parameterId))
                {
                    h5FileDataset[parameterId].CopyTo(layerParameters.Item1);
                }
            }
        }

        /// <summary>
        /// Save layer weights & bias in Tensor(s) so that they can be stored in datasets objects in a *.h5 (HDF) file
        /// </summary>
        /// <param name="originFramework">the ML Framework for which we want to save the *.h5 file
        ///     (so this file will be compatible with this Framework</param>
        // ReSharper disable once UnusedParameter.Global
        public virtual IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(NetworkSample.CompatibilityModeEnum originFramework)
        {
            var result = new Dictionary<string, CpuTensor<float>>();
            foreach (var p in Parameters)
            {
                //TODO : take into account 'originFramework'
                result.Add(p.Item2, p.Item1.ToCpuFloat());
            }
            return result;
        }

        protected string DatasetNameToDatasetPath(string datasetName)
        {
            return "/" + LayerName + "/" + LayerName + "/" + datasetName;
        }
        #endregion
        #endregion


        #region serialization

        public virtual string Serialize()
        {
            return RootSerializer().ToString();
        }
        protected Serializer RootSerializer()
        {
            var res = new Serializer()
                    .Add(nameof(Layer), GetType())
                    .Add(nameof(PreviousLayerIndexes), PreviousLayerIndexes.ToArray())
                    .Add(nameof(LayerName), LayerName)
                    .Add(nameof(Trainable), Trainable);
            return res;
        }

        public static Layer ValueOf(IDictionary<string, object> serialized, Network network)
        {
            var layerType = (string)serialized[nameof(Layer)];
            switch (layerType)
            {
                case nameof(ActivationLayer): return ActivationLayer.Deserialize(serialized, network);
                case nameof(AddLayer): return AddLayer.Deserialize(serialized, network);
                case nameof(BatchNormalizationLayer): return BatchNormalizationLayer.Deserialize(serialized, network);
                case nameof(ConcatenateLayer): return ConcatenateLayer.Deserialize(serialized, network);
                case nameof(ConvolutionLayer): return ConvolutionLayer.Deserialize(serialized, network);
                case nameof(DenseLayer): return DenseLayer.Deserialize(serialized, network);
                case nameof(DropoutLayer): return DropoutLayer.Deserialize(serialized, network);
                case nameof(EmbeddingLayer): return EmbeddingLayer.Deserialize(serialized, network);
                case nameof(FlattenLayer): return FlattenLayer.Deserialize(serialized, network);
                case nameof(InputLayer): return InputLayer.Deserialize(serialized, network);
                case nameof(LayerNormalizationLayer): return LayerNormalizationLayer.Deserialize(serialized, network);
                case nameof(LinearFunctionLayer): return LinearFunctionLayer.Deserialize(serialized, network);
                case nameof(MultiplyLayer): return MultiplyLayer.Deserialize(serialized, network);
                case nameof(NonMaxSuppressionLayer): return NonMaxSuppressionLayer.Deserialize(serialized, network);
                case nameof(PoolingLayer): return PoolingLayer.Deserialize(serialized, network);
                case nameof(PositionalEncodingAttnIsAllYouNeedLayer): return PositionalEncodingAttnIsAllYouNeedLayer.Deserialize(serialized, network);
                case nameof(RecurrentLayer): return RecurrentLayer.Deserialize(serialized, network);
                case nameof(ReshapeLayer): return ReshapeLayer.Deserialize(serialized, network);
                case nameof(ScaledDotProductAttentionLayer): return ScaledDotProductAttentionLayer.Deserialize(serialized, network);
                case nameof(SwitchSecondAndThirdDimensionLayer): return SwitchSecondAndThirdDimensionLayer.Deserialize(serialized, network);
                case nameof(UpSampling2DLayer): return UpSampling2DLayer.Deserialize(serialized, network);
                case nameof(YOLOV3Layer): return YOLOV3Layer.Deserialize(serialized, network);
                case nameof(ZeroPadding2DLayer): return ZeroPadding2DLayer.Deserialize(serialized, network);
                default: throw new NotImplementedException("don't know how to deserialize " + layerType);
            }
        }
        #endregion

        public abstract void AddToOtherNetwork(Network otherNetwork);
        protected void AddToOtherNetwork(Network otherNetwork, Func<IDictionary<string, object>, Network, Layer> deserialize)
        {
            otherNetwork.Layers.Add(deserialize(Serializer.Deserialize(Serialize()), otherNetwork));
        }

        // ReSharper disable once MemberCanBeMadeStatic.Global
        protected void Log(string msg) {Model.Log.Info(msg);}
        // ReSharper disable once MemberCanBeMadeStatic.Global
        public void LogDebug(string msg) {Model.Log.Debug(msg);}

        public int n_x
        {
            get
            {
                var result = Utils.Product(OutputShape(1));
                Debug.Assert(result>= 1);
                return result;
            }
        }
        public bool IsSigmoidActivationLayer()
        {
            return GetType() == typeof(ActivationLayer) &&((ActivationLayer) this).ActivationFunction == cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID;
        }
        public bool IsSoftmaxActivationLayer()
        {
            return GetType() == typeof(ActivationLayer) && ((ActivationLayer)this).ActivationFunction == cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX;
        }
        public bool IsInputLayer => PreviousLayerIndexes.Count == 0;
        public bool IsOutputLayer => NextLayerIndexes.Count == 0;
        public bool SameOutputShape(Layer layer)
        {
            return OutputShape(1).SequenceEqual(layer.OutputShape(1));
        }

        protected virtual string ComputeLayerName()
        {
            var result = LayerType().ToLowerInvariant();
            int countBefore = Layers.Count(l => l.LayerIndex < LayerIndex && l.LayerType() == LayerType());
            if (countBefore != 0)
            {
                result += "_" + countBefore;
            }
            return result;
        }
        public virtual string LayerType() { return GetType().Name.Replace("Layer", ""); }
        /// <summary>
        /// by default (if not overriden) output shape is the same as the previous layer
        /// </summary>
        /// <param name="batchSize"></param>
        /// <returns></returns>
        public virtual int[] OutputShape(int batchSize)
        {
            if (LazyOutputShape != null)
            {
                var result = (int[])LazyOutputShape.Clone();
                result[0] = batchSize;
                return result;
            }
            LazyOutputShape = PrevLayer.OutputShape(batchSize);
            return (int[])LazyOutputShape.Clone();
        }
        public virtual void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }
            _isDisposed = true;
            EmbeddedTensors(false).ForEach(FreeFloatTensor);
            Optimizer?.Dispose();
        }
        public override string ToString()
        {
            return LayerName + ": " + ShapeChangeDescription();
        }
        public string ContentStats()
        {
            var result = "";
            foreach (var t in EmbeddedTensors(true))
            {
                if (t == null)
                {
                    continue;
                }
                if (!string.IsNullOrEmpty(result))
                {
                    result += Environment.NewLine;
                }
                result += "t: " + t.ContentStats();
            }
            return result;
        }
        public static Layer FirstTrainableLayer(IEnumerable<Layer> layers)
        {
            foreach (var l in layers)
            {
                if (l.Trainable && l.HasParameters)
                {
                    return l;
                }
            }
            return null;
        }
        public List<Layer> PreviousLayers => PreviousLayerIndexes.Select(idx => Layers[idx]).ToList();
        public bool LayerOutputShouldBeKeptForBackwardPropagation(bool isTraining)
        {
            if (!isTraining) //if we are doing only inference 
            {
                return false; //no need to keep layer output
            }
            return LayerOutputShouldBeKeptForBackwardPropagation(true, FirstTrainableLayer(Layers));
        }
        public bool LayerOutputShouldBeKeptForBackwardPropagation(bool isTraining, Layer firstTrainableLayer)
        {
            if (!BackwardPropagationNeeded(isTraining, firstTrainableLayer))
            {
                //the BackwardPropagation method will not be called for this layer
                return false; //no need to keep in memory the layer output
            }
            if ( !OutputNeededForBackwardPropagation && NextLayers.All(l=>!l.InputNeededForBackwardPropagation) )
            {
                return false; //no need to keep in memory the layer output
            }
            return true; //we need to keep layer output in memory : it will be needed by BackwardPropagation
        }

        /// <summary>
        /// true if the BackwardPropagation method will be called for this layer
        /// </summary>
        /// <param name="isTraining"></param>
        /// <param name="firstTrainableLayer"></param>
        /// <returns></returns>
        protected bool BackwardPropagationNeeded(bool isTraining, Layer firstTrainableLayer)
        {
            if ((firstTrainableLayer == null)     //if there is no trainable layer in the network
                || !isTraining) //or if we are doing only inference 
            {
                return false; //no need to keep layer output
            }

            //if the layer is among the trainable layer
            if (LayerIndex >= firstTrainableLayer.LayerIndex)
            {
                return true; //we need to keep the layer output for backward propagation
            }

            //if the layer output is consumed by a trainable layer
            if (NextLayerIndexes.Max() >= firstTrainableLayer.LayerIndex)
            {
                return true; //we need to keep the layer output for backward propagation
            }

            return false; //no need to keep layer output in memory
        }
        protected string ShapeChangeDescription()
        {
            return Utils.ShapeToStringWithBatchSize(PrevLayer?.OutputShape(1)) + "=>" + Utils.ShapeToStringWithBatchSize(OutputShape(1));
        }
        public Layer PrevLayer => (PreviousLayerIndexes.Count == 0) ? null : Layers[PreviousLayerIndexes[0]];
        #region memory management
        protected void GetFloatTensor(ref Tensor bufferIfAny, int[] shape)
        {
            MemoryPool.GetFloatTensor(ref bufferIfAny, shape);
        }
        protected Tensor GetFloatTensor(int[] shape)
        {
            return MemoryPool.GetFloatTensor(shape);
        }
        protected void FreeFloatTensor(ref Tensor t)
        {
            MemoryPool.FreeFloatTensor(ref t);
        }
        protected void FreeFloatTensor(Tensor t)
        {
            MemoryPool.FreeFloatTensor(t);
        }
        protected Tensor GetBuffer(size_t minimalSizeInBytes)
        {
            return MemoryPool.GetBuffer(minimalSizeInBytes);
        }
        #endregion

        protected void StartForwardTimer(string key, bool isTraining)
        {
            Network.StartTimer(key, isTraining ? Network.ForwardPropagationTrainingTime : Network.ForwardPropagationInferenceTime);
        }
        protected void StopForwardTimer(string key, bool isTraining)
        {
            Network.StopTimer(key, isTraining ? Network.ForwardPropagationTrainingTime : Network.ForwardPropagationInferenceTime);
        }
        protected void StartBackwardTimer(string key)
        {
            Network.StartTimer(key, Network.BackwardPropagationTime);
        }
        protected void StopBackwardTimer(string key)
        {
            Network.StopTimer(key, Network.BackwardPropagationTime);
        }
        protected virtual List<Tensor> EmbeddedTensors(bool includeOptimizeTensors)
        {
            var result = Parameters.Select(t=>t.Item1).Concat(ParameterGradients).ToList();
            if (includeOptimizeTensors && Optimizer != null)
            {
                result.AddRange(Optimizer.EmbeddedTensors);
            }
            return result;
        }

        protected Random Rand => Network.Rand;
        protected TensorMemoryPool MemoryPool => Network.MemoryPool;
        protected List<Layer> Layers => Network.Layers;
        protected NetworkSample Sample => Network.Sample;


        protected Optimizer GetOptimizer(int[] weightShape, int[] biasShape)
        {
            return GetOptimizer(Sample.OptimizerType, weightShape, biasShape);
        }
        protected Optimizer GetOptimizer(Optimizer.OptimizationEnum optimizerType, int[] weightShape, int[] biasShape)
        {
            switch (optimizerType)
            {
                case Optimizer.OptimizationEnum.Adam:
                    if (Math.Abs(Sample.AdamW_L2Regularization) > 1e-6)
                    {
                        throw new Exception("Invalid AdamW_L2Regularization (" + Sample.AdamW_L2Regularization+") for Adam: should be 0");
                    }
                    return new Adam(MemoryPool, Sample.Adam_beta1, Sample.Adam_beta2, Sample.Adam_epsilon, 0.0, weightShape, biasShape);
                case Optimizer.OptimizationEnum.AdamW:
                    if (Math.Abs(Sample.lambdaL2Regularization) > 1e-6)
                    {
                        throw new Exception("Can't use both AdamW and L2 Regularization");
                    }
                    if (Sample.AdamW_L2Regularization < 1e-6)
                    {
                        throw new Exception("Invalid AdamW_L2Regularization (" + Sample.AdamW_L2Regularization + ") for AdamW: should be > 0");
                    }
                    return new Adam(MemoryPool, Sample.Adam_beta1, Sample.Adam_beta2, Sample.Adam_epsilon, Sample.AdamW_L2Regularization, weightShape, biasShape);
                case Optimizer.OptimizationEnum.SGD: 
                    return new Sgd(MemoryPool, Sample.SGD_momentum, Sample.SGD_usenesterov, weightShape, biasShape);
                case Optimizer.OptimizationEnum.VanillaSGDOrtho: 
                    return new VanillaSgdOrtho(MemoryPool, weightShape);
                case Optimizer.OptimizationEnum.VanillaSGD:
                default: 
                    return VanillaSgd.Instance;
            }
        }

        /// <summary>
        /// true if the layer has associated weights (or bias) to train
        /// </summary>
        private List<Layer> NextLayers => NextLayerIndexes.Select(idx => Layers[idx]).ToList();
        private bool HasParameters => Parameters.Count != 0;
    }
}
