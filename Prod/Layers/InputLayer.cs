using System;
using System.Collections.Generic;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// output shape: (batchSize, InputLayer.ChannelCount, InputLayer.H, InputLayer.Weights)
    /// </summary>
    public class InputLayer : Layer
    {
        #region Private fields
        private int ChannelCount { get; }
        private int H { get; }
        private int W { get; }
        #endregion

        public InputLayer(int channelCount, int h, int w, Network network, string layerName) : base(network, layerName)
        {
            this.ChannelCount = channelCount;
            this.H = h;
            this.W = w;
        }

        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            throw new Exception("should never call "+nameof(ForwardPropagation)+" in "+nameof(InputLayer));
        }
        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (InputLayer)b;
            var equals = true;
            equals &= Utils.Equals(ChannelCount, other.ChannelCount, id + ":ChannelCount", ref errors);
            equals &= Utils.Equals(H, other.H, id + ":H", ref errors);
            equals &= Utils.Equals(W, other.W, id + ":W", ref errors);
            return equals;
        }
        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(ChannelCount), ChannelCount).Add(nameof(H), H).Add(nameof(W), W).ToString();
        }
        public InputLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            ChannelCount = (int)serialized[nameof(ChannelCount)];
            H = (int)serialized[nameof(H)];
            W = (int)serialized[nameof(W)];
        }
        #endregion
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            throw new NotImplementedException();
        }
        public override int[] OutputShape(int batchSize) { return new[] { batchSize, ChannelCount, H, W }; }
        public override string ToString()
        {
            var result = LayerName + ": " + Utils.ShapeToStringWithBacthSize(OutputShape(1));
            result += " ("+MemoryDescription()+")";
            return result;
        }

        protected override string DefaultLayerName() { return "input_" + (1 + NbLayerOfSameTypeBefore()); }
        public override string Type() { return "InputLayer"; }
        public override void Dispose()
        {
            //do not dispose y
        }
    }
}
