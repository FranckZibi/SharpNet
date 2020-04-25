﻿using System.Collections.Generic;
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
        public AddLayer(int previousIdentityLayerIndex, int previousResidualLayerIndex, Network network, string layerName = "") : base(network, previousResidualLayerIndex, layerName)
        {
            Debug.Assert(LayerIndex>=2);
            Debug.Assert(previousIdentityLayerIndex >= 0);
            Debug.Assert(previousIdentityLayerIndex < LayerIndex);
            Debug.Assert(previousResidualLayerIndex >= 0);
            Debug.Assert(previousResidualLayerIndex < LayerIndex);
            //we add the identity shortcut connection
            AddPreviousLayer(previousIdentityLayerIndex);
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 2);
            allX[0].CopyTo(y);
            y.Update_Adding_Alpha_X(1, allX[1]);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allDx.Count == 2);
            Debug.Assert(allDx[0].SameShape(dy));
            allDx.ForEach(dy.CopyTo);
        }
        #endregion

        #region serialization
        public AddLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion

        #region clone layer
        public override Layer CloneForSlaveNetwork(Network newSlaveNetwork) { return new AddLayer(this, newSlaveNetwork); }
        private AddLayer(AddLayer toCloneFromMasterNetwork, Network newNetwork) : base(toCloneFromMasterNetwork, newNetwork) { }
        #endregion
    }
}
