using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers;

public class Flatten : Layer
{
    private readonly int start_dim;
    private readonly int end_dim;

    /// <summary>
    /// convert the input tensor to a new Tensor by aggregating all dimensions between shape[start_dim] and shape[end_dim] (included)
    /// see: https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    /// if input shape is:
    ///     (batchSize, a,b, ...., y,z)
    /// output shape will be:
    ///     (batchSize, a*b*y*z)                    if start_dim = 1 and end_dim = -1
    ///     (batchSize*a*b*y, z)                    if start_dim = 0 and end_dim = -2 ('y' index)
    /// </summary>
    /// <param name="start_dim">first dim to flatten</param>
    /// <param name="end_dim">last dim to flatten</param>
    /// <param name="network"></param>
    /// <param name="layerName"></param>
    public Flatten(int start_dim, int end_dim, Network network, string layerName = "") : base(network, layerName)
    {
        Debug.Assert(end_dim < 0 || end_dim >= start_dim);
        this.start_dim = start_dim;
        this.end_dim = end_dim;
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
            .Add(nameof(start_dim), start_dim)
            .Add(nameof(end_dim), end_dim)
            .ToString();
    }
    public static Flatten Deserialize(IDictionary<string, object> serialized, Network network)
    {
        return new Flatten(serialized.GetOrDefault(nameof(start_dim), 1), serialized.GetOrDefault(nameof(end_dim), -1), network, (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion


    #region PyTorch support
    public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
    {
        constructorLines.Add("self." + LayerName + " = torch.nn.Flatten(start_dim="+ start_dim+", end_dim=" + end_dim+ ")");
        UpdateForwardLines(forwardLines);
    }

    #endregion
    public override int[] OutputShape(int batchSize)
    {
        var inputShape = PrevLayer.OutputShape(batchSize);
        return OutputShapeAfterFlatten(inputShape, start_dim, end_dim);
    }

    public static int[] OutputShapeAfterFlatten(int[] initialShape, int start_dim, int end_dim)
    {
        if (end_dim < 0)
        {
            end_dim = initialShape.Length+end_dim;
        }
        List<int> result = new();
        for (int i = 0; i < start_dim; ++i)
        {
            result.Add(initialShape[i]);
        }

        int new_dim = 1;
        for (int i = start_dim; i <= end_dim; ++i)
        {
            new_dim *= initialShape[i];
        }
        result.Add(new_dim);
        for (int i = end_dim+1; i < initialShape.Length; ++i)
        {
            result.Add(initialShape[i]);
        }
        return result.ToArray();
    }
}