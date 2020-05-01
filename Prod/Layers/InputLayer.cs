using System;
using System.Collections.Generic;
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
        private readonly int _h;
        /// <summary>
        /// width
        /// </summary>
        private readonly int _w;
        #endregion

        public InputLayer(int c, int h, int w, Network network, string layerName) : base(network, layerName)
        {
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
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(_c), _c).Add(nameof(_h), _h).Add(nameof(_w), _w).ToString();
        }
        public InputLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _c = (int)serialized[nameof(_c)];
            _h = (int)serialized[nameof(_h)];
            _w = (int)serialized[nameof(_w)];
        }
        #endregion

        public override void AddToOtherNetwork(Network otherNetwork)
        {
            otherNetwork.Layers.Add(new InputLayer(_c, _h, _w, otherNetwork, LayerName));
        }

        public override int[] OutputShape(int batchSize) { return new[] { batchSize, _c, _h, _w }; }
        public override string ToString()
        {
            return LayerName + ": " + Utils.ShapeToStringWithBatchSize(OutputShape(1));
        }
        public override void Dispose()
        {
            //do not dispose y
        }
        public override string Type() { return "InputLayer"; }

        protected override string DefaultLayerName() { return "input_" + (1 + NbLayerOfSameTypeBefore()); }
    }
}
