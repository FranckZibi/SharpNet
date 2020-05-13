using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class ZeroPadding2DLayer : Layer
    {
        #region private fields
        private readonly int _paddingTop;
        private readonly int _paddingBottom;
        private readonly int _paddingLeft;
        private readonly int _paddingRight;
        #endregion

        public ZeroPadding2DLayer(int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int previousLayerIndex, Network network, string layerName) : base(network, new[] { previousLayerIndex}, layerName)
        {
            _paddingTop = paddingTop;
            _paddingBottom = paddingBottom;
            _paddingLeft = paddingLeft;
            _paddingRight = paddingRight;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            y.ZeroPadding(x, _paddingTop, _paddingBottom, _paddingLeft, _paddingRight);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allDx.Count == 1);
            var dx = allDx[0];
            dx.ZeroUnpadding(dy, _paddingTop, _paddingBottom, _paddingLeft, _paddingRight);
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_paddingTop), _paddingTop)
                .Add(nameof(_paddingBottom), _paddingBottom)
                .Add(nameof(_paddingLeft), _paddingLeft)
                .Add(nameof(_paddingRight), _paddingRight)
                .ToString();
        }
        public static ZeroPadding2DLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
            return new ZeroPadding2DLayer(
                (int)serialized[nameof(_paddingTop)],
                (int)serialized[nameof(_paddingBottom)],
                (int)serialized[nameof(_paddingLeft)],
                (int)serialized[nameof(_paddingRight)],
                previousLayerIndexes[0],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            var result = (int[])PrevLayer.OutputShape(batchSize).Clone();
            result[2] = _paddingTop + result[2] + _paddingBottom;
            result[3] = _paddingLeft + result[3] + _paddingRight;
            return result;
        }
    }
}