﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    public abstract class Layer : IDisposable
    {
        #region fields
        public int LayerIndex { get; }
        public string LayerName { get; }
        public List<int> NextLayerIndexes { get; } = new List<int>();
        public List<int> PreviousLayerIndexes { get; } = new List<int>();
        /// <summary>
        /// true if the layer weights should be updated during training
        /// false if it is a frozen layer
        /// </summary>
        public bool Trainable { get; set; } = true;
        protected readonly Network Network;
        private bool _isDisposed;
        private int[] _lazyOutputShape;
        #endregion

        #region constructors
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
        /// update all weights of the layer thanks to the weight (& bias)  gradients computed in the backward propagation step
        /// only used in master network
        /// </summary>
        /// <param name="batchSize"></param>
        /// <param name="learningRate"></param>
        public void UpdateWeights(int batchSize, double learningRate)
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
                Optimizer?.UpdateWeights(learningRate, batchSize, Weights, WeightGradients, Bias, BiasGradients);
            }
        }
        #endregion

        #region parameters and gradients
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
        protected virtual Optimizer Optimizer => null;

        public int TotalParams => Parameters.Select(t => t.Item1.Count).Sum();
        public virtual List<Tuple<Tensor, string>> Parameters => new List<Tuple<Tensor, string>>();
        protected virtual bool HasParameters => false;

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
        public List<Tensor> ParameterGradients
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
        public virtual void LoadParameters(IDictionary<string, Tensor> h5FileDataset, NetworkConfig.CompatibilityModeEnum originFramework)
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
        public virtual IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(NetworkConfig.CompatibilityModeEnum originFramework)
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
                case nameof(FlattenLayer): return FlattenLayer.Deserialize(serialized, network);
                case nameof(InputLayer): return InputLayer.Deserialize(serialized, network);
                case nameof(PoolingLayer): return PoolingLayer.Deserialize(serialized, network);
                case nameof(MultiplyLayer): return MultiplyLayer.Deserialize(serialized, network);
                case nameof(SimpleRnnLayer): return SimpleRnnLayer.Deserialize(serialized, network);
                case nameof(UpSampling2DLayer): return UpSampling2DLayer.Deserialize(serialized, network);
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

        public void Log(string msg)
        {
            Network.Info(msg);
        }

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
        protected virtual string DefaultLayerName() { return Type().ToLowerInvariant()+"_"+(1+NbLayerOfSameTypeBefore()); }
        public virtual string Type() { return GetType().Name.Replace("Layer", ""); }
        public virtual int ExtraElementCountForForwardPropagation(int batchSize)
        {
            return 0;
        }
        public virtual int ExtraElementCountForBackwardPropagation(int batchSize)
        {
            return 0;
        }
        
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
        public void LogContent()
        {
            Network.Info(ToString());
            foreach (var v in EmbeddedTensors(true))
            {
                Network.LogDebug(v + ": " + v.ToNumpy());
            }
            Network.Info("");
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
        public List<Layer> PreviousLayers => PreviousLayerIndexes.Select(idx => Network.Layers[idx]).ToList();
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
        
        protected string ShapeChangeDescription()
        {
            return Utils.ShapeToStringWithBatchSize(PrevLayer?.OutputShape(1)) + "=>" + Utils.ShapeToStringWithBatchSize(OutputShape(1));
        }
        protected Layer PrevLayer => (PreviousLayerIndexes.Count == 0) ? null : Network.Layers[PreviousLayerIndexes[0]];
        protected void AddPreviousLayer(int previousLayerIndex)
        {
            if (previousLayerIndex >= 0)
            {
                PreviousLayerIndexes.Add(previousLayerIndex);
                Network.Layers[previousLayerIndex].NextLayerIndexes.Add(LayerIndex);
            }
        }

        #region memory management
        protected void GetFloatTensor(ref Tensor bufferIfAny, int[] shape)
        {
            Network.MemoryPool.GetFloatTensor(ref bufferIfAny, shape);
        }
        protected Tensor GetFloatTensor(int[] shape)
        {
            return Network.MemoryPool.GetFloatTensor(shape);
        }
        protected void FreeFloatTensor(ref Tensor t)
        {
            Network.MemoryPool.FreeFloatTensor(ref t);
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
        protected bool LayerOutputShouldBeKeptForBackwardPropagation(bool isTraining)
        {
            if (!isTraining) //if we are doing only inference 
            {
                return false; //no need to keep layer output
            }
            return LayerOutputShouldBeKeptForBackwardPropagation(true, FirstTrainableLayer(Network.Layers));
        }
        //https://docs.nvidia.com/deeplearning/sdk/cudnn-archived/cudnn_701/cudnn-user-guide/index.html#cudnnBatchNormMode_t
        protected cudnnBatchNormMode_t LayerBatchNormalizationMode()
        {
            //if previous layer is a dense layer
            if (PrevLayer != null && (PrevLayer.OutputShape(1).Length == 2 || PrevLayer.IsInputLayer))
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
        /// true if the layer has associated weights (or bias) to train
        /// </summary>
        private void FreeFloatTensor(Tensor t)
        {
            Network.MemoryPool.FreeFloatTensor(t);
        }
    }
}
