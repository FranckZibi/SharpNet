using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class DropoutLayer : Layer
    {
        #region fields
        public override Tensor y { get; protected set; }        // (batchSize, C, H, W)
        private readonly double _dropProbability;
        private Tensor _dropOutMaskBufferForCpuOnly;            //only needed for CPU computation, should be null for GPU
        private readonly Random _dropOutRandomForCpuOnly = new Random(0);
        #endregion

        public DropoutLayer(double dropProbability, Network network, string layerName) : base(network, layerName)
        {
            _dropProbability = dropProbability;
        }

        public override Layer Clone(Network newNetwork) { return new DropoutLayer(this, newNetwork); }
        private DropoutLayer(DropoutLayer toClone, Network newNetwork) : base(toClone, newNetwork)
        {
            _dropProbability = toClone._dropProbability;
            _dropOutMaskBufferForCpuOnly = toClone._dropOutMaskBufferForCpuOnly?.Clone(newNetwork.GpuWrapper);
        }

        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            if (!Network.UseGPU)
            {
                _dropOutMaskBufferForCpuOnly = Network.NewNotInitializedFloatTensor(y.Shape, _dropOutMaskBufferForCpuOnly, "_DropOutMaskBufferForCpuOnly");
            }
            var x = PrevLayer.y;
            x.DropoutForward(y, _dropProbability, isTraining, _dropOutRandomForCpuOnly, _dropOutMaskBufferForCpuOnly);
        }
        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);
            var x = PrevLayer.y;
            x.DropoutBackward(dy, dx[0], _dropProbability, _dropOutMaskBufferForCpuOnly);
        }
        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (DropoutLayer)b;
            var equals = true;
            equals &= Utils.Equals(_dropProbability, other._dropProbability, epsilon, id, ref errors);
            return equals;
        }
        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(_dropProbability), _dropProbability).ToString();
        }
        public DropoutLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _dropProbability = (double) serialized[nameof(_dropProbability)];
        }
        #endregion
    }
}
