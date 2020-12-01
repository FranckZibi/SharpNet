using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public sealed class GRULayer : RecurrentLayer
    {
        #region constructor
        public GRULayer(int units, bool returnSequences, bool isBidirectional, bool trainable, Network network, string layerName) :
            base(units, cudnnRNNMode_t.CUDNN_GRU, cudnnRNNBiasMode_t.CUDNN_RNN_DOUBLE_BIAS, returnSequences, isBidirectional, trainable, network, layerName)
        {
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(HiddenSize), HiddenSize)
                .Add(nameof(_returnSequences), _returnSequences)
                .Add(nameof(IsBidirectional), IsBidirectional)
                .ToString();
        }
        public static GRULayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            Debug.Assert(network.UseGPU);
            return new GRULayer((int)serialized[nameof(HiddenSize)],
                (bool)serialized[nameof(_returnSequences)],
                (bool)serialized[nameof(IsBidirectional)],
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion
    }
}
