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
        /// <summary>
        /// pooling height
        /// -1 if we are using global pooling
        /// </summary>
        private readonly int _poolingHeight;
        /// <summary>
        /// pooling width
        /// -1 if we are using global pooling
        /// </summary>
        private readonly int _poolingWidth;
        /// <summary>
        /// pooling stride
        /// -1 if we are using global pooling
        /// </summary>
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
        public PoolingLayer(cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride, int previousLayerIndex, Network network, string layerName) : base(network, new[] { previousLayerIndex}, layerName)
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
            var x = allX[0];
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 3 || x.Dimension == 4);
            if (IsGlobalPooling(_poolingHeight))
            {
                var h = x.Shape[2];
                var w = x.Shape.Length >= 4 ? x.Shape[3] : 1;
                x.Pooling(y, _poolingMode, h, w, System.Math.Max(h, w));
                return;
            }
            x.Pooling(y, _poolingMode, _poolingHeight, _poolingWidth, _poolingStride);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);
            var x = allX[0];
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 3 || x.Dimension == 4);
            if (IsGlobalPooling(_poolingHeight))
            {
                var h = x.Shape[2];
                var w = x.Shape.Length >= 4 ? x.Shape[3] : 1;
                dy.PoolingGradient(y, x, dx[0], _poolingMode, x.Shape[2], w, System.Math.Max(h, w));
                return;
            }
            dy.PoolingGradient(y, x, dx[0], _poolingMode, _poolingHeight, _poolingWidth, _poolingStride);
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_poolingMode), (int)_poolingMode)
                .Add(nameof(_poolingHeight), _poolingHeight)
                .Add(nameof(_poolingWidth), _poolingWidth)
                .Add(nameof(_poolingStride), _poolingStride)
                .ToString();
        }
        public static PoolingLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
            return new PoolingLayer(
                (cudnnPoolingMode_t) (int) serialized[nameof(_poolingMode)],
                (int) serialized[nameof(_poolingHeight)],
                (int) serialized[nameof(_poolingWidth)],
                (int) serialized[nameof(_poolingStride)],
                previousLayerIndexes[0],
                network,
                (string) serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override string LayerType() { return IsMaxPooling(_poolingMode) ? "MaxPooling" : "AveragePooling"; }
        protected override string ComputeLayerName()
        {
            return base.ComputeLayerName().Replace("pooling", "_pooling2d_");
        }
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
            var yShape = xShape.Length == 4
                ?PoolingOutputShape4D(xShape, _poolingHeight, _poolingWidth, _poolingStride)
                :PoolingOutputShape3D(xShape, _poolingHeight, _poolingStride);
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
        public static int[] PoolingOutputShape4D(int[] inputShape, int poolingHeight, int poolingWidth, int poolingStride)
        {
            Debug.Assert(inputShape.Length == 4);
            if (IsGlobalPooling(poolingHeight))
            {
                return new[] { inputShape[0], inputShape[1], 1, 1};
            }
            var heightInput = inputShape[2];
            var heightOutput = (heightInput - poolingHeight) / poolingStride + 1;
            var widthInput = inputShape[3];
            var widthOutput = (widthInput - poolingWidth) / poolingStride + 1;
            return new[] { inputShape[0], inputShape[1], heightOutput, widthOutput };
        }


        /// <summary>
        /// Compute the pooling layer output shape given an input of shape 'inputShape'
        /// </summary>
        /// <param name="inputShape">(batchSize, x.C, heightInput)</param>
        /// <param name="poolingHeight">the pooling size</param>
        /// <param name="poolingStride">pooling stride</param>
        /// <returns>the output shape: (batchSize, x.C, y.H)</returns>
        public static int[] PoolingOutputShape3D(int[] inputShape, int poolingHeight, int poolingStride)
        {
            Debug.Assert(inputShape.Length == 3);
            if (IsGlobalPooling(poolingHeight))
            {
                return new[] { inputShape[0], inputShape[1], 1};
            }
            var heightInput = inputShape[2];
            var heightOutput = (heightInput - poolingHeight) / poolingStride + 1;
            return new[] { inputShape[0], inputShape[1], heightOutput };
        }

        private static bool IsGlobalPooling(int poolingHeight)
        {
            return (poolingHeight == -1);
        }

    }
}
