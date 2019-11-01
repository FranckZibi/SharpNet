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
        public override Tensor y { get; protected set; }

        public ConcatenateLayer(int previousLayerIndex1, int previousLayerIndex2, Network network) : base(network, previousLayerIndex1)
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
            var result = PreviousLayerIndex1.OutputShape(batchSize);
            result[1] += PreviousLayerIndex2.OutputShape(batchSize)[1];
            return result;
        }

        private Layer PreviousLayerIndex1 => PreviousLayers[0];
        private Layer PreviousLayerIndex2 => PreviousLayers[1];
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            y.Concatenate(PreviousLayerIndex1.y, PreviousLayerIndex2.y);
        }

        /// <summary>
        /// At this stage, we already know dy (output layer gradient)
        /// we want to compute PreviousLayerIndex1.dy (=dx[0]) & PreviousLayerIndex2.dy (=dx[1]) by backward propagation
        /// </summary>
        /// <param name="dy">already computed output layer gradient</param>
        /// <param name="dx">the 2 values to compute (from dy)</param>
        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
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
