using System;
using System.Collections.Generic;
using SharpNet.Data;

namespace SharpNet
{
    public class DropoutLayer : Layer
    {
        #region fields
        public override Tensor y { get; protected set; }        // (batchSize, C, H, W)
        public override Tensor dy { get; protected set; }       // same as 'y'
        private readonly double _dropProbability;
        private Tensor _dropOutMaskBufferForCpuOnly;            //only needed for CPU computation, should be null for GPU
        private readonly Random _dropOutRandomForCpuOnly = new Random(0);
        #endregion

        public DropoutLayer(double dropProbability, Network network) : base(network)
        {
            _dropProbability = dropProbability;
        }
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            if (!Network.Config.UseGPU)
            {
                _dropOutMaskBufferForCpuOnly = Network.NewTensor(y.Shape, _dropOutMaskBufferForCpuOnly, "_DropOutMaskBufferForCpuOnly");
            }
            var x = PrevLayer.y;
            x.DropoutForward(y, _dropProbability, isTraining, _dropOutRandomForCpuOnly, _dropOutMaskBufferForCpuOnly);
        }
        public override void BackwardPropagation()
        {
            //At this stage, we already know dy, we want to compute dx
            var x = PrevLayer.y;
            var dx = PrevLayer.dy;
            x.DropoutBackward(dy, dx, _dropProbability, _dropOutMaskBufferForCpuOnly);
        }

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(_dropProbability), _dropProbability).ToString();
        }
        public static DropoutLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new DropoutLayer((double)serialized[nameof(_dropProbability)], network);
        }
        #endregion
    }
}
