using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// output shape:
    ///     (batchSize, InputLayer.ChannelCount, InputLayer.H, InputLayer.Weights)
    /// </summary>
    public class InputLayer : Layer
    {
        #region Private fields
        /// <summary>
        /// channel count
        /// </summary>
        private readonly int _c;
        /// <summary>
        /// height
        /// </summary>
        private int _h;
        /// <summary>
        /// width
        /// </summary>
        private int _w;
        #endregion

        public InputLayer(int c, int h, int w, Network network, string layerName) : base(network, new int[]{}, layerName)
        {
            Debug.Assert(c>=1);
            _c = c;
            _h = h;
            _w = w;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            throw new Exception("should never call "+nameof(ForwardPropagation)+" in "+nameof(InputLayer));
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            throw new NotImplementedException();
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(_c), _c).Add(nameof(_h), _h).Add(nameof(_w), _w).ToString();
        }
        public static InputLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new InputLayer(
                (int)serialized[nameof(_c)],
                (int)serialized[nameof(_h)],
                (int)serialized[nameof(_w)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            if (_h < 0)
            {
                return new[] { batchSize, _c};
            }
            if (_w < 0)
            {
                return new[] { batchSize, _c, _h};
            }
            return new[] { batchSize, _c, _h, _w };
        }
        public override string ToString()
        {
            return LayerName + ": " + Utils.ShapeToStringWithBatchSize(OutputShape(1));
        }
        public override void Dispose()
        {
            //do not dispose y
        }
        public override string LayerType() { return "InputLayer"; }

        #region PyTorch support
        public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
        {
        }
        #endregion

        protected override string ComputeLayerName()
        {
            return base.ComputeLayerName().Replace("inputlayer", "input");
        }

        public void SetInputHeightAndWidth(int height, int width)
        {
            _h = height;
            _w = width;
            Layers.ForEach(l=>l.LazyOutputShape = null);
        }
    }
}
