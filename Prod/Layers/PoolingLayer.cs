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
    ///     (batchSize, x.C, x.H, x.W)      if 4D
    ///     (batchSize, x.H, x.W)           if 3D
    /// y shape:
    ///     (batchSize, x.C, y.H, y.W)      if 4D
    ///     (batchSize, y.H, y.W)           if 3D
    ///          y.H = (x.H − poolingHeight) / poolingStride + 1
    ///          y.W = (x.W − poolingWidth) / poolingStride + 1
    /// </summary>
    public class PoolingLayer : Layer
    {
        #region Fields
        private readonly cudnnPoolingMode_t _poolingMode;
        /// <summary>
        /// pooling height
        /// -1 if we are using global vertical pooling
        /// </summary>
        private readonly int _poolingHeight;
        /// <summary>
        /// pooling width
        /// -1 if we are using global horizontal pooling
        /// </summary>
        private readonly int _poolingWidth;
        /// <summary>
        /// vertical (height) stride
        /// -1 if we are using global vertical pooling
        /// </summary>
        private readonly int _verticalStride;
        /// <summary>
        /// horizontal (width) stride
        /// -1 if we are using global horizontal pooling
        /// </summary>
        private readonly int _horizontalStride;
        #endregion

        /// <summary>
        /// No need to configure the number of channels by filter: it is always the same as in previous layer
        /// </summary>
        /// <param name="poolingMode"></param>
        /// <param name="poolingHeight"></param>
        /// <param name="poolingWidth"></param>
        /// <param name="verticalStride"></param>
        /// <param name="horizontalStride"></param>
        /// <param name="previousLayerIndex"></param>
        /// <param name="network"></param>
        /// <param name="layerName"></param>
        public PoolingLayer(cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int previousLayerIndex, Network network, string layerName) : base(network, new[] { previousLayerIndex}, layerName)
        {
            _poolingMode = poolingMode;
            _poolingHeight = poolingHeight;
            _poolingWidth = poolingWidth;
            _verticalStride = verticalStride;
            _horizontalStride = horizontalStride;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];

            var x4D = x.Shape.Length == 4 ? x : x.Reshape(Tensor.ToPooling4D(x.Shape));
            var y4D = y.Shape.Length == 4 ? y : y.Reshape(Tensor.ToPooling4D(y.Shape));

            Debug.Assert(x4D.Dimension == y4D.Dimension);
            Debug.Assert(x4D.Dimension == 4);
            var x4DShape = x4D.Shape;

            x4D.Pooling(y4D, _poolingMode, PoolingHeight(_poolingHeight, x4DShape), PoolingWidth(_poolingWidth, x4DShape), VerticalStride(_verticalStride, x4DShape), HorizontalStride(_horizontalStride, x4DShape));
        }

        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(allDx.Count == 1);
            var x = allX[0];
            var dx = allDx[0];

            var x4D =   x.Shape.Length == 4 ? x  :  x.Reshape(Tensor.ToPooling4D(x.Shape));
            var dx4D = dx.Shape.Length == 4 ? dx : dx.Reshape(Tensor.ToPooling4D(dx.Shape));
            var y4D =   y.Shape.Length == 4 ? y  :  y.Reshape(Tensor.ToPooling4D(y.Shape));
            var dy4D = dy.Shape.Length == 4 ? dy : dy.Reshape(Tensor.ToPooling4D(dy.Shape));

            Debug.Assert(x4D.Dimension == y4D.Dimension);
            Debug.Assert(x4D.Dimension == 4);
            var x4DShape = x4D.Shape;
            dy4D.PoolingGradient(y4D, x4D, dx4D, _poolingMode, PoolingHeight(_poolingHeight, x4DShape), PoolingWidth(_poolingWidth, x4DShape), VerticalStride(_verticalStride, x4DShape), HorizontalStride(_horizontalStride, x4DShape));
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_poolingMode), (int)_poolingMode)
                .Add(nameof(_poolingHeight), _poolingHeight)
                .Add(nameof(_poolingWidth), _poolingWidth)
                .Add(nameof(_verticalStride), _verticalStride)
                .Add(nameof(_horizontalStride), _horizontalStride)
                .ToString();
        }
        public static PoolingLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];

            var verticalStride = serialized.ContainsKey("_poolingStride")?serialized["_poolingStride"]:serialized[nameof(_verticalStride)];
            var horizontalStride = serialized.ContainsKey("_poolingStride")?serialized["_poolingStride"]:serialized[nameof(_horizontalStride)];

            return new PoolingLayer(
                (cudnnPoolingMode_t) (int) serialized[nameof(_poolingMode)],
                (int) serialized[nameof(_poolingHeight)],
                (int) serialized[nameof(_poolingWidth)],
                (int)verticalStride,
                (int)horizontalStride,
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
            return LayerName +": " + ShapeChangeDescription() + " size=["+_poolingHeight+"x"+_poolingWidth+"] stride=["+_verticalStride+"x"+_horizontalStride+"]";
        }
        public override int[] OutputShape(int batchSize)
        {
            var xShape = PrevLayer.OutputShape(batchSize);
            var yShape = PoolingOutputShape(xShape, _poolingHeight, _poolingWidth, _verticalStride, _horizontalStride);
            Debug.Assert(yShape.Min() >= 1);
            return yShape;
        }
        /// <summary>
        /// Compute the pooling layer output shape given an input of shape 'inputShape'
        /// </summary>
        /// <param name="inputShape">
        ///     (batchSize, x.C, heightInput, widthInput)   if 4D
        ///     (batchSize, heightInput, widthInput)        if 3D
        /// </param>
        /// <param name="poolingHeight">the pooling size is (poolingHeight, poolingWidth)</param>
        /// <param name="poolingWidth">the pooling size is (poolingHeight, poolingWidth)</param>
        /// <param name="verticalStride">pooling vertical stride</param>
        /// <param name="horizontalStride">pooling horizontal stride</param>
        /// <returns>
        /// the output shape:
        ///     (batchSize, x.C, y.H, y.W)                  if 4D
        ///     (batchSize, y.H, y.W)                       if 3D
        /// </returns>
        public static int[] PoolingOutputShape(int[] inputShape, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            Debug.Assert(inputShape.Length >= 3);

            var outputShape = (int[]) inputShape.Clone();

            //we compute the output height: it is the second from last element in outputShape array
            outputShape[^2] = (inputShape[^2] - PoolingHeight(poolingHeight, inputShape)) / VerticalStride(verticalStride, inputShape) + 1;

            //we compute the output width: it is the last element in outputShape array
            outputShape[^1] = (inputShape[^1] - PoolingWidth(poolingWidth, inputShape)) / HorizontalStride(horizontalStride, inputShape) + 1;

            return outputShape;
        }

        private static int PoolingHeight(int defaultPoolingHeight, int[] inputShape) { return (defaultPoolingHeight == -1) ? inputShape[^2] : defaultPoolingHeight; }
        private static int VerticalStride(int defaultVerticalStride, int[] inputShape) { return (defaultVerticalStride == -1) ? inputShape[^2] : defaultVerticalStride; }
        private static int PoolingWidth(int defaultPoolingWidth, int[] inputShape) { return (defaultPoolingWidth == -1) ? inputShape[^1] : defaultPoolingWidth; }
        private static int HorizontalStride(int defaultHorizontalStride, int[] inputShape) { return (defaultHorizontalStride == -1) ? inputShape[^1] : defaultHorizontalStride; }
    }
}
