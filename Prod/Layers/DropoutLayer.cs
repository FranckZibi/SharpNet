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
        private Tensor _dropOutMaskBufferForCpu;                            //only needed for Cpu (null for GPU)
        private Tensor _randomNumberGeneratorStatesBufferForGPU;            //only needed for GPU (null for Cpu)
        private Tensor _dropoutReserveSpaceForGPU;                          //only needed for GPU (null for Cpu)
        private IntPtr _dropoutDescriptorForGPU = IntPtr.Zero;              //only needed for GPU (null for Cpu)
        private readonly Random _dropOutRandomForCpuOnly = new Random(0);
        #endregion

        public DropoutLayer(double dropProbability, Network network, string layerName) : base(network, layerName)
        {
            _dropProbability = dropProbability;
        }
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            if (!Network.UseGPU)
            {
                GetNotInitializedFloatTensor(ref _dropOutMaskBufferForCpu, y.Shape, "_DropOutMaskBufferForCpuOnly");
            }
            allX[0].DropoutForward(y, _dropProbability, isTraining, _dropOutRandomForCpuOnly, _dropOutMaskBufferForCpu, ref _randomNumberGeneratorStatesBufferForGPU, ref _dropoutReserveSpaceForGPU, ref _dropoutDescriptorForGPU, Network.MemoryPool);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);
            allX[0].DropoutBackward(dy, dx[0], _dropProbability, _dropOutMaskBufferForCpu, _randomNumberGeneratorStatesBufferForGPU, _dropoutReserveSpaceForGPU, _dropoutDescriptorForGPU);
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
            _randomNumberGeneratorStatesBufferForGPU?.Dispose();
            _dropoutReserveSpaceForGPU?.Dispose();

            //unmanaged memory
            if (_dropoutDescriptorForGPU != IntPtr.Zero)
            {
                var res = CudnnWrapper.cudnnDestroyDropoutDescriptor(_dropoutDescriptorForGPU);
                GPUWrapper.CheckStatus(res, ToString);
            }
            _randomNumberGeneratorStatesBufferForGPU = null;
            _dropoutReserveSpaceForGPU = null;
            _dropoutDescriptorForGPU = IntPtr.Zero;
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
