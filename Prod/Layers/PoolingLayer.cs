using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// x     (batchSize, x.C, x.H, x.W)
    ///
    /// y     (batchSize, x.C, y.H, y.W)
    ///          y.H = (x.H−poolingSize) / poolingStride + 1
    ///          y.W = (x.W−poolingSize) / poolingStride + 1
    /// </summary>
    public class PoolingLayer : Layer
    {
        #region Fields
        private readonly cudnnPoolingMode_t _poolingMode;
        private readonly int _poolingHeight;
        private readonly int _poolingWidth;
        private readonly int _poolingStride;
        public override Tensor y { get; protected set; }
        #endregion

        /// <summary>
        /// No need to configure the number of channels by filter: it is always the same as in previous layer
        /// </summary>
        /// <param name="poolingMode"></param>
        /// <param name="poolingHeight"></param>
        /// <param name="poolingWidth"></param>
        /// <param name="poolingStride"></param>
        /// <param name="previousLayerIndex"></param>
        /// <param name="network"></param>
        public PoolingLayer(cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride, int previousLayerIndex, Network network, string layerName) : base(network, previousLayerIndex, layerName)
        {
            _poolingMode = poolingMode;
            _poolingHeight = poolingHeight;
            _poolingWidth = poolingWidth;
            _poolingStride = poolingStride;
        }

        public override Layer Clone(Network newNetwork) { return new PoolingLayer(this, newNetwork); }
        private PoolingLayer(PoolingLayer toClone, Network newNetwork) : base(toClone, newNetwork)
        {
            _poolingMode = toClone._poolingMode;
            _poolingHeight = toClone._poolingHeight;
            _poolingWidth = toClone._poolingWidth;
            _poolingStride = toClone._poolingStride;
        }

        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PrevLayer.y;
            x.Pooling(y, _poolingMode, _poolingHeight, _poolingWidth, _poolingStride);
        }
        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);
            var x = PrevLayer.y;
            dy.PoolingGradient(y, x, dx[0], _poolingMode, _poolingHeight, _poolingWidth, _poolingStride);
        }

     
        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (PoolingLayer)b;
            var equals = true;
            equals &= Utils.Equals(_poolingMode, other._poolingMode, id + ":_poolingMode", ref errors);
            equals &= Utils.Equals(_poolingHeight, other._poolingHeight, id + ":_poolingHeight", ref errors);
            equals &= Utils.Equals(_poolingWidth, other._poolingWidth, id + ":_poolingWidth", ref errors);
            equals &= Utils.Equals(_poolingStride, other._poolingStride, id + ":_poolingStride", ref errors);
            return equals;
        }
        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_poolingHeight), _poolingHeight).Add(nameof(_poolingWidth), _poolingWidth)
                .Add(nameof(_poolingStride), _poolingStride)
                .Add(nameof(_poolingMode), (int)_poolingMode)
                .ToString();
        }
        public PoolingLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _poolingMode = (cudnnPoolingMode_t)(int)serialized[nameof(_poolingMode)];
            _poolingHeight = (int)serialized[nameof(_poolingHeight)];
            _poolingWidth = (int)serialized[nameof(_poolingWidth)];
            _poolingStride = (int)serialized[nameof(_poolingStride)];
        }
        #endregion
        protected override string DefaultLayerName()
        {
            return (IsMaxPooling(_poolingMode) ? "max_pooling2d_" : "average_pooling2d_") + (1 + NbLayerOfSameTypeBefore());
        }
        public override string Type() { return IsMaxPooling(_poolingMode) ? "MaxPooling" : "AveragePooling"; }
        public static bool IsMaxPooling(cudnnPoolingMode_t poolingMode)
        {
            return poolingMode == cudnnPoolingMode_t.CUDNN_POOLING_MAX ||
                   poolingMode == cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC;
        }
        public override string ToString()
        {
            var result = LayerName +": " + ShapeChangeDescription() + " size=["+_poolingHeight+"x"+_poolingWidth+"] stride="+_poolingStride;
            result += " (" + MemoryDescription() + ")";
            return result;
        }
        public override int[] OutputShape(int batchSize)
        {
            var xShape = PrevLayer.OutputShape(batchSize);
            var yShape = PoolingOutputShape(xShape, _poolingHeight, _poolingWidth, _poolingStride);
            Debug.Assert(yShape.Min() >= 1);
            return yShape;
        }

        /// <summary>
        /// Compute the pooling layer output shape given an input of shape 'shapeInput'
        /// </summary>
        /// <param name="shapeInput">(batchSize, x.C, heightInput, widthInput)</param>
        /// <param name="poolingHeight">the pooling size is (poolingHeight, poolingWidth)</param>
        /// <param name="poolingWidth">the pooling size is (poolingHeight, poolingWidth)</param>
        /// <param name="poolingStride">pooling stride</param>
        /// <returns>the output shape: (batchSize, x.C, y.H, y.W)</returns>
        public static int[] PoolingOutputShape(int[] shapeInput, int poolingHeight, int poolingWidth, int poolingStride)
        {
            Debug.Assert(shapeInput.Length == 4);
            Debug.Assert(poolingStride >= 1);
            var batchSize = shapeInput[0];
            var heightInput = shapeInput[2];
            var widthInput = shapeInput[3];
            var heightOutput = (heightInput - poolingHeight) / poolingStride + 1;
            var widthOutput = (widthInput - poolingWidth) / poolingStride + 1;
            return new[] { batchSize, shapeInput[1], heightOutput, widthOutput };
        }
    }
}
