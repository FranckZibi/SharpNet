using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;

namespace SharpNet
{
    public abstract class Layer : IDisposable
    {
        #region fields
        public Tensor dyIdentityConnection { get; set; }
        public int LayerIndex { get; }
        protected readonly Network Network;
        private readonly List<int> _previousLayerIndexes = new List<int>();
        private readonly List<int> _nextLayerIndexes = new List<int>();
        #endregion

        protected Layer(Network network) :this(network, network.Layers.Count-1) 
        {
        }
        protected Layer(Network network, int previousLayerIndex)
        {
            Network = network;
            LayerIndex = network.Layers.Count;
            AddPreviousLayer(previousLayerIndex);
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
        public List<Layer> NextLayers => _nextLayerIndexes.Select(idx => Network.Layers[idx]).ToList();
        public bool IsInputLayer => _previousLayerIndexes.Count == 0;
        public bool IsOutputLayer => _nextLayerIndexes.Count == 0;
        public bool SameOutputShape(Layer layer)
        {
            return OutputShape(10).SequenceEqual(layer.OutputShape(10));
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
        public static bool IsValidYSet(Tensor data)
        {
            Debug.Assert(!data.UseGPU);
            if (data.UseDoublePrecision)
            {
                return data.AsDoubleCpuContent.All(IsValidY);
            }
            return data.AsFloatCpuContent.All(x=> IsValidY(x));
        }
        public virtual ulong BytesByBatchSize => (ulong)(2 * Utils.Product(OutputShape(1)) * Network.Config.TypeSize); //y dy
        public virtual string SummaryName() { return GetType().Name.Replace("Layer","");}
        public ulong BytesIndependantOfBatchSize => Tensor.OccupiedMemoryInBytes(TensorsIndependantOfBatchSize);
        public abstract void ForwardPropagation(bool isTraining);
        public abstract void BackwardPropagation();
        public virtual void UpdateWeights(double learningRate) { }
        public abstract Tensor y { get; protected set; } //output of layer 
        // ReSharper disable once InconsistentNaming
        //gradient of layer output (= null if it is the input layer)
        public abstract Tensor dy { get; protected set; } 
        //by default (if not overriden) output shape is the same as the previous layer
        public virtual int[] OutputShape(int batchSize) { return PrevLayer.OutputShape(batchSize); }
        public virtual int TotalParams => 0;
        public virtual void DisableBias() {}
        public virtual void Dispose()
        {
            EmbeddedTensors.ForEach(x => x?.Dispose());
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
        public abstract string Serialize();
        protected Serializer RootSerializer()
        {
            return new Serializer().Add(nameof(Layer), GetType())
                    .Add(nameof(LayerIndex), LayerIndex)
                    .Add(nameof(_previousLayerIndexes), _previousLayerIndexes.ToArray())
                    .Add(nameof(_nextLayerIndexes), _nextLayerIndexes.ToArray())
                ;
        }
        protected Layer(IDictionary<string, object> serialized, Network network) : this(network, -1)
        {
            LayerIndex = (int)serialized[nameof(LayerIndex)];
            _previousLayerIndexes = ((int[])serialized[nameof(_previousLayerIndexes)]).ToList();
            _nextLayerIndexes = ((int[])serialized[nameof(_nextLayerIndexes)]).ToList();
        }
        public static Layer ValueOf(IDictionary<string, object> serialized, Network network)
        {
            var layerType = (string)serialized[nameof(Layer)];
            switch (layerType)
            {
                case nameof(ActivationLayer): return ActivationLayer.Deserialize(serialized, network);
                case nameof(BatchNormalizationLayer): return BatchNormalizationLayer.Deserialize(serialized, network);
                case nameof(ConvolutionLayer): return ConvolutionLayer.Deserialize(serialized, network);
                case nameof(DenseLayer): return DenseLayer.Deserialize(serialized, network);
                case nameof(DropoutLayer): return DropoutLayer.Deserialize(serialized, network);
                case nameof(InputLayer): return InputLayer.Deserialize(serialized, network);
                case nameof(PoolingLayer): return PoolingLayer.Deserialize(serialized, network);
                case nameof(FlattenLayer): return FlattenLayer.Deserialize(network);
                case nameof(SumLayer): return new SumLayer(serialized, network);
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
        protected void Update_dy_With_GradientFromShortcutIdentityConnection()
        {
            if (NextLayers.Count >= 2)
            {
                Debug.Assert(dyIdentityConnection != null);
                dy.Update_Adding_Alpha_X(1.0, dyIdentityConnection);
            }
        }
        protected void Allocate_y_dy_if_necessary()
        {
            var batchSize = PrevLayer.y.Shape[0];
            var outputShape = OutputShape(batchSize);
            if (y == null || !y.Shape.SequenceEqual(outputShape))
            {
                y = Network.NewTensor(outputShape, y, nameof(y));
            }
            if (dy == null || !dy.Shape.SequenceEqual(outputShape))
            {
                dy = Network.NewTensor(outputShape, dy, nameof(dy));
                if (NextLayers.Count >= 2)
                {
                    dyIdentityConnection = Network.NewTensor(outputShape, dyIdentityConnection, nameof(dyIdentityConnection));
                }
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
        protected List<Layer> PreviousLayers => _previousLayerIndexes.Select(idx => Network.Layers[idx]).ToList();
        protected Layer NextLayer => (_nextLayerIndexes.Count == 0) ? null : Network.Layers[_nextLayerIndexes[0]];
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
                var result = new List<Tensor> { y, dy };
                result.RemoveAll(t => t == null);
                return result;
            }
        }
        private static bool IsValidY(double x)
        {
            return Math.Abs(x) <= 1e-9 || Math.Abs(x - 1.0) <= 1e-9;
        }
        private ulong OccupiedMemoryInBytes => Tensor.OccupiedMemoryInBytes(EmbeddedTensors);
    }
}
