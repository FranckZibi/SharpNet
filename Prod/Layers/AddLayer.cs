using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// used for Residual Network
    /// Layer that is the sum of the Previous Layer and the Shortcut Layer
    /// </summary>
    public class AddLayer : Layer
    {
        public AddLayer(int[] previousResidualLayerIndexes, Network network, string layerName = "") : base(network, previousResidualLayerIndexes, layerName)
        {
        }
        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 2);
            allX[0].CopyTo(y);
            y.Update_Adding_Alpha_X(1, allX[1]);
        }
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(allDx.Count == 2);
            Debug.Assert(allDx[0].SameShape(dy));
            allDx.ForEach(dy.CopyTo);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;
        #endregion

        #region serialization

        public static AddLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new AddLayer((int[])serialized[nameof(PreviousLayerIndexes)], network, (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion
    }
}
