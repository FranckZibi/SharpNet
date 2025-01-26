using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers;


// TODO : implement layer
public class StochasticDepthLayer : Layer
{
    private readonly double p;
    private readonly string mode;

    /// <summary>
    /// Apply StochasticDepth, see: https://pytorch.org/vision/main/generated/torchvision.ops.StochasticDepth.html
    /// </summary>
    /// <param name="p">proba to apply Stochastic Depth</param>
    /// <param name="mode"></param>
    /// <param name="network"></param>
    /// <param name="layerName"></param>
    public StochasticDepthLayer(double p, string mode, Network network, string layerName = "") : base(network, layerName)
    {
        this.p = p;
        this.mode = mode;
    }

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 1);
        allX[0].CopyTo(y);
    }
    public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> dx)
    {
        Debug.Assert(allX_NotUsed.Count == 0);
        Debug.Assert(y_NotUsed == null);
        Debug.Assert(dx.Count == 1);
        if (PrevLayer.IsInputLayer)
        {
            return;
        }
        dy.CopyTo(dx[0]);
    }
    public override bool OutputNeededForBackwardPropagation => false;
    public override bool InputNeededForBackwardPropagation => false;

    #endregion

    #region serialization
    public override string Serialize()
    {
        return RootSerializer()
            .Add(nameof(p), p)
            .Add(nameof(mode), mode)
            .ToString();
    }
    public static StochasticDepthLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        return new StochasticDepthLayer( (double)serialized[nameof(p)], (string)serialized[nameof(mode)], network, (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion


    #region PyTorch support
    public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
    {
        constructorLines.Add("self." + LayerName + " = torch.nn.StochasticDepth(p=" +p+ ", mode='" + mode+ "')");
        UpdateForwardLines(forwardLines);
    }

    #endregion
    public override int[] OutputShape(int batchSize)
    {
        return PrevLayer.OutputShape(batchSize);
    }
}
