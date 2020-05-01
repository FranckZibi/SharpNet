﻿using System;
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

            x.DropoutForward(y, _dropProbability, isTraining, _dropOutRandomForCpuOnly, _dropoutReservedSpaceForTraining, Network.MemoryPool);
            if (!LayerOutputShouldBeKeptForBackwardPropagation(isTraining))
            {
                FreeFloatTensor(ref _dropoutReservedSpaceForTraining);
            }

        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);
            Debug.Assert(_dropoutReservedSpaceForTraining != null);
            Debug.Assert(_dropoutReservedSpaceForTraining.UseGPU == y.UseGPU);
            allX[0].DropoutBackward(dy, dx[0], _dropProbability, _dropoutReservedSpaceForTraining);
            FreeFloatTensor(ref _dropoutReservedSpaceForTraining);
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(_dropProbability), _dropProbability).ToString();
        }
        public DropoutLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _dropProbability = (double)serialized[nameof(_dropProbability)];
        }
        #endregion

        public override void AddToOtherNetwork(Network otherNetwork)
        {
            otherNetwork.Layers.Add(new DropoutLayer(_dropProbability,otherNetwork, LayerName));
        }

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
                Network.MemoryPool.GetBuffer(ref _dropoutReservedSpaceForTraining, dropoutReservedSpaceInBytes, nameof(_dropoutReservedSpaceForTraining));
            }
            else
            {
                GetFloatTensor(ref _dropoutReservedSpaceForTraining, x.Shape);
            }
            Debug.Assert(_dropoutReservedSpaceForTraining.UseGPU == x.UseGPU);
        }
    }
}
