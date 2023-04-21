using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers;

public class FlattenLayer : Layer
{
    private readonly bool _flattenInputTensorOnLastDimension;

    /// <summary>
    /// convert the input tensor to a 2D Tensor
    /// if input shape is:
    ///     (batchSize, a,b, ...., y,z)
    /// output shape will be:
    ///     (batchSize, a*b*y*z)                    if flattenInputTensorOnLastDimension == false
    ///     (batchSize*a*b*y, z)                    if flattenInputTensorOnLastDimension == true
    /// </summary>
    /// <param name="flattenInputTensorOnLastDimension"></param>
    /// <param name="network"></param>
    /// <param name="layerName"></param>
    public FlattenLayer(bool flattenInputTensorOnLastDimension, Network network, string layerName = "") : base(network, layerName)
    {
        _flattenInputTensorOnLastDimension = flattenInputTensorOnLastDimension;
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
            .Add(nameof(_flattenInputTensorOnLastDimension), _flattenInputTensorOnLastDimension)
            .ToString();
    }
    public static FlattenLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        var flattenInputTensorOnLastDimension = serialized.GetOrDefault(nameof(_flattenInputTensorOnLastDimension), false);
        return new FlattenLayer(flattenInputTensorOnLastDimension, network);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion

    public override int[] OutputShape(int batchSize)
    {
        var inputShape = PrevLayer.OutputShape(batchSize);
        var count = Utils.Product(inputShape);
        return _flattenInputTensorOnLastDimension 
            ? new[] { count / inputShape[^1], inputShape[^1] } 
            : new[] { inputShape[0], count / inputShape[0] };
    }
}