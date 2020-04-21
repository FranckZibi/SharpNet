using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public abstract class Layer : IDisposable
    {
        #region fields
        public int LayerIndex { get; }
        public string LayerName { get; }
        protected readonly Network Network;
        private int[] _lazyOutputShape;
        public List<int> NextLayerIndexes { get; } = new List<int>();
        public List<int> PreviousLayerIndexes { get; } = new List<int>();
        #endregion

        /// <summary>
        /// Clone constructor
        /// </summary>
        /// <param name="toClone">the layer to be cloned</param>
        /// <param name="newNetwork">the network where the cloned layer will be located</param>
        protected Layer(Layer toClone, Network newNetwork)
        {
            Network = newNetwork;
            LayerIndex = toClone.LayerIndex;
            LayerName = toClone.LayerName;
            PreviousLayerIndexes.Clear();
            PreviousLayerIndexes.AddRange(toClone.PreviousLayerIndexes);
            NextLayerIndexes.Clear();
            NextLayerIndexes.AddRange(toClone.NextLayerIndexes);
            if (_lazyOutputShape != null)
            {
                _lazyOutputShape = (int[])toClone._lazyOutputShape.Clone();
            }
        }

        protected Layer(Network network, int previousLayerIndex, string layerName)
        {
            Network = network;
            LayerIndex = network.Layers.Count;
            // ReSharper disable once VirtualMemberCallInConstructor
            LayerName = string.IsNullOrEmpty(layerName) ? DefaultLayerName() : layerName;
            AddPreviousLayer(previousLayerIndex);
            network.OnLayerAddOrRemove();
        }
        protected Layer(Network network, string layerName) : this(network, network.Layers.Count - 1, layerName)
        {
        }


        /// <summary>
        ///  At this stage, we already know 'x', we want to compute 'y'
        /// </summary>
        /// <param name="allX"></param>
        /// <param name="y"></param>
        /// <param name="isTraining">true if we are currently training the network
        ///     false if we are just using it to make a prediction </param>
        public abstract void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining);
        /// <summary>
        ///  At this stage, we already know 'x'+'y'+'dy', we want to compute 'dx' by backward propagation
        /// </summary>
        /// <param name="allX">[in] all layer inputs</param>
        /// <param name="y">[in] the already computed output</param>
        /// <param name="dy">[in] the already computed output gradient</param>
        /// <param name="dx">[out] input gradient (dx) to compute from the output gradient (dy)</param>
        public abstract void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx);
        public virtual void UpdateWeights(int batchSize, double learningRate) { }
        public virtual void ResetWeights(bool resetAlsoOptimizerWeights = true) { }

        public abstract Layer Clone(Network newNetwork);
        /// <summary>
        /// compares the current layer with the other layer 'b' 
        /// </summary>
        /// <param name="b">2nd Layer to compare</param>
        /// <param name="epsilon">ignore difference between numeric values if less then epsilon </param>
        /// <param name="id"></param>
        /// <param name="errors"></param>
        /// <returns>true if a difference was observed, false if same layers</returns>
        public virtual bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            bool equals = true;
            id += ":" + LayerName;
            equals &= Utils.Equals(LayerIndex, b.LayerIndex, id+":LayerIndex", ref errors);
            equals &= Utils.Equals(LayerName, b.LayerName, id+":LayerName", ref errors);
            equals &= Utils.Equals(GetType(), b.GetType(), id+ ":GetType", ref errors);
            equals &= Utils.EqualsList(PreviousLayerIndexes, b.PreviousLayerIndexes, id+ ":PreviousLayerIndexes", ref errors);
            equals &= Utils.EqualsList(NextLayerIndexes, b.NextLayerIndexes, id+ ":NextLayerIndexes", ref errors);
            equals &= Utils.EqualsList(TrainableTensorsIndependentOfBatchSize, b.TrainableTensorsIndependentOfBatchSize, epsilon, id+ ":TrainableTensorsIndependentOfBatchSize", ref errors);
            return equals;
        }
        protected int NbLayerOfSameTypeBefore()
        {
            int result = 0;
            for (var layerIndex = 0; layerIndex < LayerIndex; ++layerIndex)
            {
                if (Network.Layers[layerIndex].GetType() == GetType())
                {
                    ++result;
                }
            }
            return result;
        }

        /// <summary>
        /// true if the layer weights should be updated during training
        /// false if it is a frozen layer
        /// </summary>
        public bool Trainable { get; set; } = true;

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
        public bool IsInputLayer => PreviousLayerIndexes.Count == 0;
        public bool IsOutputLayer => NextLayerIndexes.Count == 0;
        public bool SameOutputShape(Layer layer)
        {
            return OutputShape(1).SequenceEqual(layer.OutputShape(1));
        }
        public void CheckConsistency()
        {
            if (!Tensor.AreCompatible(EmbeddedTensors))
            {
                throw new Exception("invalid tensor consistency");
            }
        }
        protected virtual string DefaultLayerName() { return Type().ToLowerInvariant()+"_"+(1+NbLayerOfSameTypeBefore()); }
        public virtual string Type() { return GetType().Name.Replace("Layer", ""); }
        public ulong BytesIndependentOfBatchSize => Tensor.OccupiedMemoryInBytes(TensorsIndependentOfBatchSize);

        public virtual int ExtraElementCountForForwardPropagation(int batchSize)
        {
            return 0;
        }
        public virtual int ExtraElementCountForBackwardPropagation(int batchSize)
        {
            return 0;
        }
        
        #region *.h5 file (HDF) management

        /// <summary>
        /// Initialize layer weights & bias from datasets objects extracted from a *.h5 (HDF) file
        /// </summary>
        /// <param name="h5FileDataset">all datasets objects in the *.h5 file</param>
        /// <param name="originFramework">the ML Framework from where the *.h5 file comes from</param>
        public virtual void LoadFromH5Dataset(Dictionary<string, Tensor> h5FileDataset, NetworkConfig.CompatibilityModeEnum originFramework)
        {
        }

        /// <summary>
        /// Save layer weights & bias in Tensor(s) so that they can be stored in datasets objects in a *.h5 (HDF) file
        /// </summary>
        /// <param name="h5FileDataset">all datasets objects in the *.h5 file</param>
        /// <param name="originFramework">the ML Framework for which we want to save the *.h5 file
        /// (so this file will be compatible with this Framework</param>
        public virtual void SaveToH5Dataset(List<Tuple<string, Tensor>> h5FileDataset, NetworkConfig.CompatibilityModeEnum originFramework)
        {
            throw new NotImplementedException(); //TODO
        }

        protected string DatasetNameToDatasetPath(string datasetName)
        {
            return "/" + LayerName + "/" + LayerName + "/" + datasetName;
        }
        #endregion


        /// <summary>
        /// by default (if not overriden) output shape is the same as the previous layer
        /// </summary>
        /// <param name="batchSize"></param>
        /// <returns></returns>
        public virtual int[] OutputShape(int batchSize)
        {
            if (_lazyOutputShape != null)
            {
                var result = (int[]) _lazyOutputShape.Clone();
                result[0] = batchSize;
                return result;
            }
            _lazyOutputShape = PrevLayer.OutputShape(batchSize);
            return (int[])_lazyOutputShape.Clone();
        }
        public virtual int DisableBias()
        {
            return PreviousLayers.Select(l => l.DisableBias()).Sum();
        }
        public virtual void Dispose()
        {
            EmbeddedTensors.ForEach(x => x?.Dispose());
            _lazyOutputShape = null;
        }
        public void LogContent()
        {
            Network.Config.Logger.Info(ToString());
            foreach (var v in EmbeddedTensors)
            {
                Network.Config.Logger.Debug(v + ": " + v.ToNumpy());
            }
            Network.Config.Logger.Info("");
        }
        public override string ToString()
        {
            return LayerName + ": " + ShapeChangeDescription() + " ("+ MemoryDescription()+")";
        }

        public int TotalParams => TrainableTensorsIndependentOfBatchSize.Select(t => t.Count).Sum();
        public List<Tensor> TensorsIndependentOfBatchSize => TrainableTensorsIndependentOfBatchSize.Concat(NonTrainableTensorsIndependentOfBatchSize).ToList();
        protected virtual List<Tensor> TrainableTensorsIndependentOfBatchSize => new List<Tensor>();
        protected virtual List<Tensor> NonTrainableTensorsIndependentOfBatchSize => new List<Tensor>();

        public string ContentStats()
        {
            var result = "";
            foreach (var t in EmbeddedTensors)
            {
                if (t == null)
                {
                    continue;
                }
                if (!string.IsNullOrEmpty(result))
                {
                    result += Environment.NewLine;
                }
                result += t.Description+": " + t.ContentStats();
            }
            return result;
        }

        #region serialization
        public virtual string Serialize()
        {
            return RootSerializer().ToString();
        }
        protected Serializer RootSerializer()
        {
            var res = new Serializer().Add(nameof(Layer), GetType())
                    .Add(nameof(LayerIndex), LayerIndex)
                    .Add(nameof(LayerName), LayerName)
                    .Add(nameof(PreviousLayerIndexes), PreviousLayerIndexes.ToArray())
                    .Add(nameof(NextLayerIndexes), NextLayerIndexes.ToArray())
                ;
            //we serialize all trainable tensors (weights)
            foreach (var trainableTensor in TrainableTensorsIndependentOfBatchSize)
            {
                res = res.Add(trainableTensor);
            }
            return res;
        }
        protected Layer(IDictionary<string, object> serialized, Network network)
        {
            Network = network;
            LayerIndex = (int)serialized[nameof(LayerIndex)];
            LayerName = (string)serialized[nameof(LayerName)];
            PreviousLayerIndexes = ((int[])serialized[nameof(PreviousLayerIndexes)]).ToList();
            NextLayerIndexes = ((int[])serialized[nameof(NextLayerIndexes)]).ToList();
        }
        public static Layer ValueOf(IDictionary<string, object> serialized, Network network)
        {
            var layerType = (string)serialized[nameof(Layer)];
            switch (layerType)
            {
                case nameof(ActivationLayer): return new ActivationLayer(serialized, network);
                case nameof(AddLayer): return new AddLayer(serialized, network);
                case nameof(BatchNormalizationLayer): return new BatchNormalizationLayer(serialized, network);
                case nameof(ConcatenateLayer): return new ConcatenateLayer(serialized, network);
                case nameof(ConvolutionLayer): return new ConvolutionLayer(serialized, network);
                case nameof(DenseLayer): return new DenseLayer(serialized, network);
                case nameof(DropoutLayer): return new DropoutLayer(serialized, network);
                case nameof(FlattenLayer): return new FlattenLayer(serialized, network);
                case nameof(InputLayer): return new InputLayer(serialized, network);
                case nameof(PoolingLayer): return new PoolingLayer(serialized, network);
                case nameof(MultiplyLayer): return new MultiplyLayer(serialized, network);
                case nameof(SimpleRnnLayer): return new SimpleRnnLayer(serialized, network);
                default: throw new NotImplementedException("don't know how to deserialize " + layerType);
            }
        }
        #endregion

        //https://docs.nvidia.com/deeplearning/sdk/cudnn-archived/cudnn_701/cudnn-user-guide/index.html#cudnnBatchNormMode_t
        protected cudnnBatchNormMode_t LayerBatchNormalizationMode()
        {
            //if previous layer is a dense layer
            if (PrevLayer != null && (PrevLayer.OutputShape(1).Length == 2 || PrevLayer.IsInputLayer) )
            {
                //Normalization is performed per-activation. 
                //This mode is intended to be used after non-convolutional network layers. 
                //In this mode bnBias and bnScale tensor dimensions are (1, C, H, W)
                return cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION;
            }
            //Normalization is performed over N + spatial dimensions.
            //This mode is intended for use after convolutional layers(where spatial invariance is desired).
            //In this mode bnBias, bnScale tensor dimensions are (1, C, 1, 1)
            return cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL;
        }

        #region those 4 methods are only used for NonReg tests
        /// <summary>
        /// the weight if any (used only for tests)
        /// </summary>
        public virtual Tensor Weights => null;
        /// <summary>
        /// the weight gradient if any (used only for tests)
        /// </summary>
        public virtual Tensor WeightGradients => null;
        /// <summary>
        /// the bias if any (used only for tests)
        /// </summary>
        public virtual Tensor Bias => null;
        /// <summary>
        /// the bias gradient if any (used only for tests)
        /// </summary>
        public virtual Tensor BiasGradients => null;
        #endregion

        protected string MemoryDescription()
        {
            return TotalParams != 0 ? (TotalParams + " neurons") : "";
        }
        protected string ShapeChangeDescription()
        {
            return Utils.ShapeToStringWithBatchSize(PrevLayer?.OutputShape(1)) + "=>" + Utils.ShapeToStringWithBatchSize(OutputShape(1));
        }
        protected List<Tensor> EmbeddedTensors
        {
            get
            {
                var result = TensorsIndependentOfBatchSize;
                result.AddRange(TensorsDependentOfBatchSize);
                result.RemoveAll(t => t == null);
                return result;
            }
        }
        protected Layer PrevLayer => (PreviousLayerIndexes.Count == 0) ? null : Network.Layers[PreviousLayerIndexes[0]];
        public List<Layer> PreviousLayers => PreviousLayerIndexes.Select(idx => Network.Layers[idx]).ToList();
        protected void AddPreviousLayer(int previousLayerIndex)
        {
            if (previousLayerIndex >= 0)
            {
                PreviousLayerIndexes.Add(previousLayerIndex);
                Network.Layers[previousLayerIndex].NextLayerIndexes.Add(LayerIndex);
            }
        }
        protected void GetNotInitializedFloatTensor(ref Tensor bufferIfAny, int[] shape, string description)
        {
            Network.MemoryPool.GetNotInitializedFloatTensor(ref bufferIfAny, shape, description);
        }

        protected void FreeMemory(ref Tensor t)
        {
            Network.MemoryPool.FreeMemory(ref t);
        }
     
        protected Tensor GetNotInitializedFloatTensor(int[] shape, string description)
        {
            return Network.MemoryPool.GetNotInitializedFloatTensor(shape, description);
        }

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

        protected bool LayerOutputShouldBeKeptForBackwardPropagation(bool isTraining)
        {
            if (!isTraining) //if we are doing only inference 
            {
                return false; //no need to keep layer output
            }
            return LayerOutputShouldBeKeptForBackwardPropagation(true, FirstTrainableLayer(Network.Layers));
        }

        public bool LayerOutputShouldBeKeptForBackwardPropagation(bool isTraining, Layer firstTrainableLayer)
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

        public static Layer FirstTrainableLayer(IEnumerable<Layer> layers)
        {
            foreach (var l in layers)
            {
                if (l.Trainable && l.TotalParams > 0)
                {
                    return l;
                }
            }
            return null;
        }

        protected virtual List<Tensor> TensorsDependentOfBatchSize => new List<Tensor>();
    }
}
