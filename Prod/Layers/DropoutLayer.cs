using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class DropoutLayer : Layer
    {
        #region fields
        private readonly double _dropoutRate;
        private Tensor _dropoutReservedSpaceForTraining;
        #endregion

        public DropoutLayer(double dropoutRate, Network network, string layerName) : base(network, layerName)
        {
            _dropoutRate = dropoutRate;
        }

        #region forward and backward propagation
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
                FreeFloatTensor(ref _dropoutReservedSpaceForTraining);
            }
            x.DropoutForward(y, _dropoutRate, isTraining, Rand, _dropoutReservedSpaceForTraining);
            if (!BackwardPropagationNeeded(isTraining, FirstTrainableLayer(Layers)))
            {
                FreeFloatTensor(ref _dropoutReservedSpaceForTraining);
            }
        }

        public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(dx.Count == 1);
            Debug.Assert(_dropoutReservedSpaceForTraining != null);
            allX[0].DropoutBackward(dy, dx[0], _dropoutRate, _dropoutReservedSpaceForTraining);
            FreeFloatTensor(ref _dropoutReservedSpaceForTraining);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(_dropoutRate), _dropoutRate).ToString();
        }
        public static DropoutLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var dropoutRate = serialized.ContainsKey("_dropProbability") 
                    ? (double)serialized["_dropProbability"]
                    :(double)serialized[nameof(_dropoutRate)];
            return new DropoutLayer(dropoutRate, network, (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion


        #region PyTorch support
        public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
        {
            //see: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
            constructorLines.Add("self." + LayerName + " = torch.nn.Dropout(p=" + _dropoutRate.ToString(CultureInfo.InvariantCulture) + ", inplace=False)");
            UpdateForwardLines(forwardLines);
        }
        #endregion

        public override void Dispose()
        {
            base.Dispose();
            //managed memory
            FreeFloatTensor(ref _dropoutReservedSpaceForTraining);
        }

        private void InitializeDropoutReservedSpaceForTraining(Tensor x)
        {
            if (x.UseGPU)
            {
                var xDesc = Network.GpuWrapper.TensorDesc(cudnnDataType_t.CUDNN_DATA_FLOAT, x.Shape);
                var res = CudnnWrapper.cudnnDropoutGetReserveSpaceSize(xDesc, out var dropoutReservedSpaceInBytes);
                GPUWrapper.CheckStatus(res);
                MemoryPool.GetBuffer(ref _dropoutReservedSpaceForTraining, dropoutReservedSpaceInBytes);
            }
            else
            {
                GetFloatTensor(ref _dropoutReservedSpaceForTraining, x.Shape);
            }
            Debug.Assert(_dropoutReservedSpaceForTraining.UseGPU == x.UseGPU);
        }
    }
}
