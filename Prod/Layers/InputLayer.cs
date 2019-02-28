﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            throw new Exception("should never call"+nameof(ForwardPropagation)+" in "+nameof(InputLayer)); ;
        }

        public override string Serialize()
        {
            return RootSerializer().Add(nameof(ChannelCount), ChannelCount).Add(nameof(H), H).Add(nameof(W), W).ToString();
        }
        public static InputLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new InputLayer((int)serialized[nameof(ChannelCount)], (int)serialized[nameof(H)], (int)serialized[nameof(W)], network);
        }

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
        public override void Dispose()
        {
            //do not dipose y
        }
        //do not take into account 'dy' (only y)
        public override ulong BytesByBatchSize => (ulong) (Utils.Product(OutputShape(1)) * Network.Config.TypeSize);
    }
}
