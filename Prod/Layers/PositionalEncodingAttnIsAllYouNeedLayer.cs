using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers;

/// <summary>
/// Add Positional Encoding according to Paper : 'Attention Is All You Need'
/// </summary>
public class PositionalEncodingAttnIsAllYouNeedLayer : Layer
{

    public const int DEFAULT_N_POSITIONAL_ENCODING = 10000;
    
    private readonly int N;

    public PositionalEncodingAttnIsAllYouNeedLayer(int n, Network network, string layerName = "") : base(network, layerName)
    {
        N = n;
    }
    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 1);
        allX[0].CopyTo(y);
        y.UpdateWithPositionalEncoding_AttnIsAllYouNeed(N);
    }
    public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        Debug.Assert(allX_NotUsed.Count == 0);
        Debug.Assert(y_NotUsed == null);
        Debug.Assert(allDx.Count == 1);
        Debug.Assert(allDx[0].SameShape(dy));
        dy.CopyTo(allDx[0]);
    }
    public override bool OutputNeededForBackwardPropagation => false;
    public override bool InputNeededForBackwardPropagation => false;
    #endregion

    #region serialization
    public override string Serialize()
    {
        return RootSerializer()
            .Add(nameof(N), N)
            .ToString();
    }
    public static PositionalEncodingAttnIsAllYouNeedLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        return new PositionalEncodingAttnIsAllYouNeedLayer(
            (int)serialized[nameof(N)],
            network,
            (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion
}