using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;

namespace SharpNet
{
    //used for Residual Network
    //Layer that is the sum of the Previous Layer and the Shortcut Layer
    public class AddLayer : Layer
    {
        public override Tensor y { get; protected set; }

        public AddLayer(int previousIdentityLayerIndex, int previousResidualLayerIndex, Network network) : base(network, previousResidualLayerIndex)
        {
            Debug.Assert(LayerIndex>=2);
            Debug.Assert(previousIdentityLayerIndex >= 0);
            Debug.Assert(previousResidualLayerIndex >= 0);
            //we add the identity shortcut connection
            AddPreviousLayer(previousIdentityLayerIndex);
        }
        public override Layer Clone(Network newNetwork) {return new AddLayer(this, newNetwork);}
        private AddLayer(AddLayer toClone, Network newNetwork) : base(toClone, newNetwork) {}

        #region serialization
        public AddLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion

        private Layer PreviousResidualLayer => PreviousLayers[0];
        private Layer PreviousIdentityLayer => PreviousLayers[1];
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PreviousResidualLayer.y;
            x.CopyTo(y);
            y.Update_Adding_Alpha_X(1.0, PreviousIdentityLayer.y);
        }
        public override void BackwardPropagation(Tensor dy, List<Tensor> allDx)
        {
            allDx.ForEach(dy.CopyTo);
        }
    }
}
