using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers;

public class ReshapeLayer : Layer
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

    public ReshapeLayer(int c, int h, int w, Network network, string layerName) : base(network, layerName)
    {
        Debug.Assert(c >= 1);
        _c = c;
        _h = h;
        _w = w;
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
        return RootSerializer().Add(nameof(_c), _c).Add(nameof(_h), _h).Add(nameof(_w), _w).ToString();
    }
    public static ReshapeLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        return new ReshapeLayer(
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
        List<int> newShape = new() { batchSize, _c };
        if (_h != -1)
        {
            newShape.Add(_h);
            if (_w != -1)
            {
                newShape.Add(_w);
            }
        }
        return newShape.ToArray();
    }
}