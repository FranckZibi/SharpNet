﻿using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;

namespace SharpNet
{
    //used for Dense Network
    //Layer that is the concatenation of 2 previous layers
    public class ConcatenateLayer : Layer
    {
        public override Tensor y { get; protected set; }
        public override Tensor dy { get; protected set; }

        public ConcatenateLayer(int previousLayerIndex1, int previousLayerIndex2, Network network) : base(network, previousLayerIndex1)
        {
            Debug.Assert(LayerIndex>=2);
            Debug.Assert(previousLayerIndex1 >= 0);
            Debug.Assert(previousLayerIndex2 >= 0);
            //we add the identity shortcut connection
            AddPreviousLayer(previousLayerIndex2);
        }

        #region serialization
        public ConcatenateLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion
        public override int[] OutputShape(int batchSize)
        {
            // the number of channels is the sum of the the 2 previous layers
            var result = PreviousLayerIndex1.OutputShape(batchSize);
            result[1] += PreviousLayerIndex2.OutputShape(batchSize)[1];
            return result;
        }

        private Layer PreviousLayerIndex1 => PreviousLayers[0];
        private Layer PreviousLayerIndex2 => PreviousLayers[1];
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            y.Concatenate(PreviousLayerIndex1.y, PreviousLayerIndex2.y);
        }
        public override void BackwardPropagation()
        {
            //At this stage, we already know dy
            //we want to compute PreviousLayerIndex1.dy & PreviousLayerIndex2.dy by backward propagation
            dy.Split(PreviousLayerIndex1.dy, PreviousLayerIndex2.dy);
        }
    }
}