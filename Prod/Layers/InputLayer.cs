using System;
using System.Collections.Generic;
using SharpNet.Data;

namespace SharpNet
{
    public class InputLayer : Layer
    {
        #region Private fields
        private int ChannelCount { get; }
        private int H { get; }
        private int W { get; }
        public override Tensor y { get; protected set; }            // (batchSize, InputLayer.ChannelCount, InputLayer.H, InputLayer.Weights)
        public override Tensor dy {get => null; protected set => throw new Exception("no dy available in input layer");}
        #endregion

        public InputLayer(int channelCount, int h, int w, Network network) : base(network)
        {
            this.ChannelCount = channelCount;
            this.H = h;
            this.W = w;
        }
        public override void ForwardPropagation(bool isTraining)
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
            var allAreOk = true;
            allAreOk &= Utils.Equals(ChannelCount, other.ChannelCount, id + ":ChannelCount", ref errors);
            allAreOk &= Utils.Equals(H, other.H, id + ":H", ref errors);
            allAreOk &= Utils.Equals(W, other.W, id + ":W", ref errors);
            return allAreOk;
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
        public override void BackwardPropagation()
        {
            throw new NotImplementedException();
        }
        public override int[] OutputShape(int batchSize) { return new[] { batchSize, ChannelCount, H, W }; }
        public override string ToString()
        {
            var result = SummaryName() + ": " + Utils.ShapeToStringWithBacthSize(OutputShape(1));
            result += " ("+MemoryDescription()+")";
            return result;
        }

        public override string SummaryName() { return "input_" + (1 + NbLayerOfSameTypeBefore()); }
        public override string Type() { return "InputLayer"; }
        public override void Dispose()
        {
            //do not dipose y
        }
        //do not take into account 'dy' (only y)
        public override ulong BytesByBatchSize => (ulong) (Utils.Product(OutputShape(1)) * Network.Config.TypeSize);
    }
}
