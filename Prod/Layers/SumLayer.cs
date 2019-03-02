﻿using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;

namespace SharpNet
{
    //used for Residual Network
    //Layer that is the sum of the Previous Layer and the Shortcut Layer
    public class SumLayer : Layer
    {
        public override Tensor y { get; protected set; }
        public override Tensor dy { get; protected set; }

        public SumLayer(int previousIdentityLayerIndex, int previousResidualLayerIndex, Network network) : base(network, previousResidualLayerIndex)
        {
            Debug.Assert(LayerIndex>=2);
            Debug.Assert(previousIdentityLayerIndex >= 0);
            Debug.Assert(previousResidualLayerIndex >= 0);
            //we add the identity shortcut connection
            AddPreviousLayer(previousIdentityLayerIndex);
        }

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .ToString();
        }
        public SumLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion

        private Layer PreviousResidualLayer => PreviousLayers[0];
        private Layer PreviousIdentityLayer => PreviousLayers[1];
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            var x = PreviousResidualLayer.y;
            x.CopyTo(y);
            y.Update_Adding_Alpha_X(1.0, PreviousIdentityLayer.y);
        }
        public override void BackwardPropagation()
        {
            Debug.Assert(y.SameShape(dy));
            dy.CopyTo(PreviousResidualLayer.dy);
            if (PreviousIdentityLayer.NextLayers.Count == 1)
            {
                //previous layer is a convolution layer used to change dimension
                dy.CopyTo(PreviousIdentityLayer.dy);
            }
            else
            {
                //direct identity shortcut between previous layer and current layer (because they have the same diemension)
                dy.CopyTo(PreviousIdentityLayer.dyIdentityConnection);
            }
        }
    }
}
