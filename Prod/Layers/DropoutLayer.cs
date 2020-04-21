using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class DropoutLayer : Layer
    {
        #region fields
        private readonly double _dropProbability;
        private readonly Random _dropOutRandomForCpuOnly = new Random(0);
        private Tensor _dropoutReservedSpaceForTraining;
        #endregion

        public DropoutLayer(double dropProbability, Network network, string layerName) : base(network, layerName)
        {
            _dropProbability = dropProbability;
        }

        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];

            if (isTraining)
            {
                //we initialize the dropout reserved space buffer
                InitializeDropoutReservedSpaceForTraining(x);
            }
            else
            {
                //no need of dropout reserved space for inference
                FreeMemory(ref _dropoutReservedSpaceForTraining);
            }

            x.DropoutForward(y, _dropProbability, isTraining, _dropOutRandomForCpuOnly, _dropoutReservedSpaceForTraining, Network.MemoryPool);
            if (!LayerOutputShouldBeKeptForBackwardPropagation(isTraining))
            {
                FreeMemory(ref _dropoutReservedSpaceForTraining);
            }

        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);
            Debug.Assert(_dropoutReservedSpaceForTraining != null);
            Debug.Assert(_dropoutReservedSpaceForTraining.UseGPU == y.UseGPU);
            allX[0].DropoutBackward(dy, dx[0], _dropProbability, _dropoutReservedSpaceForTraining);
            FreeMemory(ref _dropoutReservedSpaceForTraining);
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

        public override void Dispose()
        {
            base.Dispose();
            //managed memory
            FreeMemory(ref _dropoutReservedSpaceForTraining);
        }

        public override Layer Clone(Network newNetwork) { return new DropoutLayer(this, newNetwork); }
        private DropoutLayer(DropoutLayer toClone, Network newNetwork) : base(toClone, newNetwork)
       {
           _dropProbability = toClone._dropProbability;
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

        private void InitializeDropoutReservedSpaceForTraining(Tensor x)
        {
            if (x.UseGPU)
            {
                var xDesc = Network.GpuWrapper.TensorDesc(cudnnDataType_t.CUDNN_DATA_FLOAT, x.Shape);
                var res = CudnnWrapper.cudnnDropoutGetReserveSpaceSize(xDesc, out var dropoutReservedSpaceInBytes);
                GPUWrapper.CheckStatus(res);
                Network.MemoryPool.GetBuffer(ref _dropoutReservedSpaceForTraining, dropoutReservedSpaceInBytes, nameof(_dropoutReservedSpaceForTraining));
            }
            else
            {
                GetNotInitializedFloatTensor(ref _dropoutReservedSpaceForTraining, x.Shape, nameof(_dropoutReservedSpaceForTraining));
            }
            Debug.Assert(_dropoutReservedSpaceForTraining.UseGPU == x.UseGPU);
        }


    }
}
