using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// used for Dense Network
    /// Layer that is the concatenation of 2 previous layers
    /// </summary>
    public class ConcatenateLayer : Layer
    {
        public ConcatenateLayer(int[] previousLayers, Network network, string layerName) : base(network, previousLayers, layerName)
        {
            Debug.Assert(LayerIndex >= 2);
            Debug.Assert(previousLayers.Length >= 2);
            Debug.Assert(previousLayers.Min()>= 0);
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count >= 2);
            y.Concatenate(allX);
        }

        /// <summary>
        /// At this stage, we already know dy (output layer gradient)
        /// we want to compute PreviousLayerIndex1.dy (=dx[0]) & PreviousLayerIndex2.dy (=dx[1]) by backward propagation
        /// </summary>
        /// <param name="y_NotUsed"></param>
        /// <param name="dy">already computed output layer gradient</param>
        /// <param name="dx">the 2 values to compute (from dy)</param>
        /// <param name="allX_NotUsed"></param>
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(dx.Count >= 2);
            dy.Split(dx);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;
        #endregion

        #region serialization
        public static ConcatenateLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
            return new ConcatenateLayer(previousLayerIndexes, network, (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            var result = PreviousLayers[0].OutputShape(batchSize);
            // the number of channels is the sum of channels of all previous layers
            result[1] = PreviousLayers.Select(l => l.OutputShape(1)[1]).Sum();
            return result;
        }
    }
}
