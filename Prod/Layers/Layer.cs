﻿using System;
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
        protected readonly Network Network;
        private readonly List<int> _previousLayerIndexes = new List<int>();
        private readonly List<int> _nextLayerIndexes = new List<int>();
        private int[] _lazyOutputShape;
        #endregion

        #region public properties
        public abstract Tensor y { get; protected set; } //output of layer 
        // ReSharper disable once InconsistentNaming
        //gradient of layer output (= null if it is the input layer)
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
            _previousLayerIndexes.Clear();
            _previousLayerIndexes.AddRange(toClone._previousLayerIndexes);
            _nextLayerIndexes.Clear();
            _nextLayerIndexes.AddRange(toClone._nextLayerIndexes);
            if (_lazyOutputShape != null)
            {
                _lazyOutputShape = (int[])toClone._lazyOutputShape.Clone();
            }
        }

        protected Layer(Network network) :this(network, network.Layers.Count-1) 
        {
        }
        protected Layer(Network network, int previousLayerIndex)
        {
            Network = network;
            LayerIndex = network.Layers.Count;
            AddPreviousLayer(previousLayerIndex);
        }

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
            equals &= Utils.Equals(SummaryName(), b.SummaryName(), id + ":SummaryName", ref errors);
            id += ":" + SummaryName();
            equals &= Utils.Equals(LayerIndex, b.LayerIndex, id+":LayerIndex", ref errors);
            equals &= Utils.Equals(GetType(), b.GetType(), id+ ":GetType", ref errors);
            equals &= Utils.EqualsList(_previousLayerIndexes, b._previousLayerIndexes, id+ ":_previousLayerIndexes", ref errors);
            equals &= Utils.EqualsList(_nextLayerIndexes, b._nextLayerIndexes, id+ ":_nextLayerIndexes", ref errors);
            equals &= Utils.EqualsList(TensorsIndependantOfBatchSize, b.TensorsIndependantOfBatchSize, epsilon, id+ ":TensorsIndependantOfBatchSize", ref errors);
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
        public bool IsInputLayer => _previousLayerIndexes.Count == 0;
        public bool IsOutputLayer => _nextLayerIndexes.Count == 0;
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
        public void Set_y(Tensor yValue)
        {
            y = yValue;
        }
      
        public ulong BytesByBatchSize => (ulong)(Utils.Product(OutputShape(1)) * Network.Config.TypeSize); //y
        public virtual string SummaryName() { return Type().ToLowerInvariant()+"_"+(1+NbLayerOfSameTypeBefore()); }
        public virtual string Type() { return GetType().Name.Replace("Layer", ""); }
        public ulong BytesIndependantOfBatchSize => Tensor.OccupiedMemoryInBytes(TensorsIndependantOfBatchSize);
        public abstract void ForwardPropagation(bool isTraining);
        //At this stage, we already know dy, we want to compute dx by backward propagation
        public abstract void BackwardPropagation(Tensor dy, List<Tensor> dx);
        public virtual void UpdateWeights(double learningRate) { }
        public virtual void ResetWeights(bool resetAlsoOptimizerWeights = true) { }


        //by default (if not overriden) output shape is the same as the previous layer
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
        public virtual int TotalParams => 0;

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
                Network.Config.Logger.Info(v + ": " + v.ToNumpy());
            }
            Network.Config.Logger.Info("");
        }
        public override string ToString()
        {
            return SummaryName() + ": " + ShapeChangeDescription() + " ("+ MemoryDescription()+")";
        }
        public virtual List<Tensor> TensorsIndependantOfBatchSize => new List<Tensor>();

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
            return new Serializer().Add(nameof(Layer), GetType())
                    .Add(nameof(LayerIndex), LayerIndex)
                    .Add(nameof(_previousLayerIndexes), _previousLayerIndexes.ToArray())
                    .Add(nameof(_nextLayerIndexes), _nextLayerIndexes.ToArray())
                ;
        }
        protected Layer(IDictionary<string, object> serialized, Network network)
        {
            Network = network;
            LayerIndex = (int)serialized[nameof(LayerIndex)];
            _previousLayerIndexes = ((int[])serialized[nameof(_previousLayerIndexes)]).ToList();
            _nextLayerIndexes = ((int[])serialized[nameof(_nextLayerIndexes)]).ToList();
        }
        public static Layer ValueOf(IDictionary<string, object> serialized, Network network)
        {
            var layerType = (string)serialized[nameof(Layer)];
            switch (layerType)
            {
                case nameof(ActivationLayer): return new ActivationLayer(serialized, network);
                case nameof(BatchNormalizationLayer): return new BatchNormalizationLayer(serialized, network);
                case nameof(ConvolutionLayer): return new ConvolutionLayer(serialized, network);
                case nameof(DenseLayer): return new DenseLayer(serialized, network);
                case nameof(DropoutLayer): return new DropoutLayer(serialized, network);
                case nameof(InputLayer): return new InputLayer(serialized, network);
                case nameof(PoolingLayer): return new PoolingLayer(serialized, network);
                case nameof(FlattenLayer): return new FlattenLayer(serialized, network);
                case nameof(AddLayer): return new AddLayer(serialized, network);
                case nameof(ConcatenateLayer): return new ConcatenateLayer(serialized, network);
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
        protected virtual void Allocate_y_if_necessary()
        {
            var batchSize = PrevLayer.y.Shape[0];
            var outputShape = OutputShape(batchSize);
            if (y == null || !y.Shape.SequenceEqual(outputShape))
            {
                y = Network.NewNotInitializedTensor(outputShape, y, nameof(y));
            }
        }
        protected string MemoryDescription()
        {
            var result = "";
            if (TotalParams != 0)
            {
                result += TotalParams + " neurons / ";
            }
            result += Utils.MemoryBytesToString(OccupiedMemoryInBytes) + ": ";
            result += Utils.MemoryBytesToString(BytesByBatchSize) + "/batchSize+" + Utils.MemoryBytesToString(BytesIndependantOfBatchSize);
            return result;
        }
        protected string ShapeChangeDescription()
        {
            return Utils.ShapeToStringWithBacthSize(PrevLayer?.OutputShape(1)) + "=>" + Utils.ShapeToStringWithBacthSize(OutputShape(1));
        }
        protected List<Tensor> EmbeddedTensors
        {
            get
            {
                var result = TensorsIndependantOfBatchSize;
                result.AddRange(TensorsDependantOfBatchSize);
                result.RemoveAll(t => t == null);
                return result;
            }
        }
        protected Layer PrevLayer => (_previousLayerIndexes.Count == 0) ? null : Network.Layers[_previousLayerIndexes[0]];
        public List<Layer> PreviousLayers => _previousLayerIndexes.Select(idx => Network.Layers[idx]).ToList();
        protected void AddPreviousLayer(int previousLayerIndex)
        {
            if (previousLayerIndex >= 0)
            {
                _previousLayerIndexes.Add(previousLayerIndex);
                Network.Layers[previousLayerIndex]._nextLayerIndexes.Add(LayerIndex);
            }
        }

        private List<Tensor> TensorsDependantOfBatchSize
        {
            get
            {
                var result = new List<Tensor> {y};
                result.RemoveAll(t => t == null);
                return result;
            }
        }
      
        private ulong OccupiedMemoryInBytes => Tensor.OccupiedMemoryInBytes(EmbeddedTensors);
    }
}
