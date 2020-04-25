using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// x shape:
    ///     (batchSize, x.C, x.H, x.W)
    /// y shape:
    ///     (batchSize, x.C, y.H, y.W)
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
        /// <param name="layerName"></param>
        public PoolingLayer(cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride, int previousLayerIndex, Network network, string layerName) : base(network, previousLayerIndex, layerName)
        {
            _poolingMode = poolingMode;
            _poolingHeight = poolingHeight;
            _poolingWidth = poolingWidth;
            _poolingStride = poolingStride;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            allX[0].Pooling(y, _poolingMode, _poolingHeight, _poolingWidth, _poolingStride);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);
            dy.PoolingGradient(y, allX[0], dx[0], _poolingMode, _poolingHeight, _poolingWidth, _poolingStride);
        }
        #endregion

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

        #region layer clone
        public override Layer CloneForSlaveNetwork(Network newSlaveNetwork) { return new PoolingLayer(this, newSlaveNetwork); }
        private PoolingLayer(PoolingLayer toCloneFromMasterNetwork, Network newNetwork) : base(toCloneFromMasterNetwork, newNetwork)
        {
            _poolingMode = toCloneFromMasterNetwork._poolingMode;
            _poolingHeight = toCloneFromMasterNetwork._poolingHeight;
            _poolingWidth = toCloneFromMasterNetwork._poolingWidth;
            _poolingStride = toCloneFromMasterNetwork._poolingStride;
        }
        #endregion

        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (PoolingLayer)b;
            var equals = true;
            equals &= Utils.Equals(_poolingMode, other._poolingMode, id + nameof(_poolingMode), ref errors);
            equals &= Utils.Equals(_poolingHeight, other._poolingHeight, id + nameof(_poolingHeight), ref errors);
            equals &= Utils.Equals(_poolingWidth, other._poolingWidth, id + nameof(_poolingWidth), ref errors);
            equals &= Utils.Equals(_poolingStride, other._poolingStride, id + nameof(_poolingStride), ref errors);
            return equals;
        }
        public override string Type() { return IsMaxPooling(_poolingMode) ? "MaxPooling" : "AveragePooling"; }
        public static bool IsMaxPooling(cudnnPoolingMode_t poolingMode)
        {
            return poolingMode == cudnnPoolingMode_t.CUDNN_POOLING_MAX ||
                   poolingMode == cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC;
        }
        public override string ToString()
        {
            return LayerName +": " + ShapeChangeDescription() + " size=["+_poolingHeight+"x"+_poolingWidth+"] stride="+_poolingStride;
        }
        public override int[] OutputShape(int batchSize)
        {
            var xShape = PrevLayer.OutputShape(batchSize);
            var yShape = PoolingOutputShape(xShape, _poolingHeight, _poolingWidth, _poolingStride);
            Debug.Assert(yShape.Min() >= 1);
            return yShape;
        }
        /// <summary>
        /// Compute the pooling layer output shape given an input of shape 'inputShape'
        /// </summary>
        /// <param name="inputShape">(batchSize, x.C, heightInput, widthInput)</param>
        /// <param name="poolingHeight">the pooling size is (poolingHeight, poolingWidth)</param>
        /// <param name="poolingWidth">the pooling size is (poolingHeight, poolingWidth)</param>
        /// <param name="poolingStride">pooling stride</param>
        /// <returns>the output shape: (batchSize, x.C, y.H, y.W)</returns>
        public static int[] PoolingOutputShape(int[] inputShape, int poolingHeight, int poolingWidth, int poolingStride)
        {
            Debug.Assert(inputShape.Length == 4);
            Debug.Assert(poolingStride >= 1);
            var batchSize = inputShape[0];
            var heightInput = inputShape[2];
            var widthInput = inputShape[3];
            var heightOutput = (heightInput - poolingHeight) / poolingStride + 1;
            var widthOutput = (widthInput - poolingWidth) / poolingStride + 1;
            return new[] { batchSize, inputShape[1], heightOutput, widthOutput };
        }

        protected override string DefaultLayerName()
        {
            return (IsMaxPooling(_poolingMode) ? "max_pooling2d_" : "average_pooling2d_") + (1 + NbLayerOfSameTypeBefore());
        }
    }
}
