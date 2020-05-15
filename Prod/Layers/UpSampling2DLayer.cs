using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class UpSampling2DLayer : Layer
    {
        public enum InterpolationEnum {Nearest, Bilinear};

        #region private fields
        private readonly int _rowFactor;
        private readonly int _colFactor;
        private readonly InterpolationEnum _interpolation;

        #endregion

        public UpSampling2DLayer(int rowFactor, int colFactor, InterpolationEnum interpolation, Network network, string layerName) : base(network, layerName)
        {
            Debug.Assert(LayerIndex >= 2);
            _rowFactor = rowFactor;
            _colFactor = colFactor;
            _interpolation = interpolation;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            y.UpSampling2D(x, _rowFactor, _colFactor, _interpolation);
        }
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(allDx.Count == 1);
            var dx = allDx[0];
            dx.DownSampling2D(dy, _rowFactor, _colFactor);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_rowFactor), _rowFactor)
                .Add(nameof(_colFactor), _colFactor)
                .Add(nameof(_interpolation), (int)_interpolation)
                .ToString();
        }
        public static UpSampling2DLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new UpSampling2DLayer(
                (int)serialized[nameof(_rowFactor)],
                (int)serialized[nameof(_colFactor)],
                (InterpolationEnum)serialized[nameof(_interpolation)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            var result = (int[])PrevLayer.OutputShape(batchSize).Clone();
            result[2] *= _rowFactor;
            result[3] *= _colFactor;
            return result;
        }
    }
}