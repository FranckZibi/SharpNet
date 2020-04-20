using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    //used for Dense Network
    //Layer that is the concatenation of 2 previous layers
    public class ConcatenateLayer : Layer
    {
        public ConcatenateLayer(int previousLayerIndex1, int previousLayerIndex2, Network network, string layerName) : base(network, previousLayerIndex1, layerName)
        {
            Debug.Assert(LayerIndex>=2);
            Debug.Assert(previousLayerIndex1 >= 0);
            Debug.Assert(previousLayerIndex2 >= 0);
            AddPreviousLayer(previousLayerIndex2);
        }

        public override Layer Clone(Network newNetwork) { return new ConcatenateLayer(this, newNetwork); }
        private ConcatenateLayer(ConcatenateLayer toClone, Network newNetwork) : base(toClone, newNetwork) { }

        #region serialization
        public ConcatenateLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            // the number of channels is the sum of the 2 previous layers
            var result = PreviousLayers[0].OutputShape(batchSize);
            result[1] += PreviousLayers[1].OutputShape(batchSize)[1];
            return result;
        }

        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 2);
            y.Concatenate(allX[0], allX[1]);
        }

        /// <summary>
        /// At this stage, we already know dy (output layer gradient)
        /// we want to compute PreviousLayerIndex1.dy (=dx[0]) & PreviousLayerIndex2.dy (=dx[1]) by backward propagation
        /// </summary>
        /// <param name="allX"></param>
        /// <param name="y"></param>
        /// <param name="dy">already computed output layer gradient</param>
        /// <param name="dx">the 2 values to compute (from dy)</param>
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 2);
            var dx0 = dx[0];
            var dx1 = dx[1];
            if (ReferenceEquals(dx0, dx1))
            {
                throw new Exception("the same buffer has been used twice in " + this);
            }
            dy.Split(dx0, dx1);
        }
    }
}
