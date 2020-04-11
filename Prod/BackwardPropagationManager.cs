using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet
{
    public class BackwardPropagationManager : IDisposable
    {
        #region Private fields
        private readonly Network _network;
        private readonly List<Tensor> _availableTensorOrderedByCount = new List<Tensor>();
        private const string MockKey = "Mock";
        private ulong _bytesByBatchSizeForGradientComputation;
        private Tensor _dyOfLastLayer;
        #endregion

        public BackwardPropagationManager(Network network)
        {
            this._network = network;
        }

        /// <summary>
        /// the needed memory in bytes to compute all gradients for a mini batch size of '1'
        /// </summary>
        public ulong BytesByBatchSizeForGradientComputation
        {
            get
            {
                if (_bytesByBatchSizeForGradientComputation > 0 || _network.Layers.Count == 0)
                {
                    return _bytesByBatchSizeForGradientComputation;
                }
                var tmpdyOfLastLayer = _dyOfLastLayer;
                var tmpCache = new List<Tensor>(_availableTensorOrderedByCount);
                _availableTensorOrderedByCount.Clear();
                _dyOfLastLayer = NewNotInitializedFloatTensor(_network.Layers.Last().OutputShape(1), true);
                BackwardPropagation();
                _bytesByBatchSizeForGradientComputation = _availableTensorOrderedByCount.Select(CapacityInBytes).Sum() + CapacityInBytes(_dyOfLastLayer);
                _dyOfLastLayer = tmpdyOfLastLayer;
                _availableTensorOrderedByCount.Clear();
                _availableTensorOrderedByCount.AddRange(tmpCache);
                return _bytesByBatchSizeForGradientComputation;
            }
        }
        public Tensor dyOfLastLayer
        {
            get
            {
                var shape = (_network.Layers.Last().y != null)
                    ? _network.Layers.Last().y.Shape
                    : _network.Layers.Last().OutputShape(1);
                _dyOfLastLayer = _network.NewNotInitializedFloatTensor(shape, _dyOfLastLayer, nameof(dyOfLastLayer));
                return _dyOfLastLayer;
            }
        }
        public void BackwardPropagation()
        {
            bool isMock = IsMock(_dyOfLastLayer);
            var layerIndex_to_dY = new Tensor[_network.Layers.Count];
            layerIndex_to_dY[_network.Layers.Count - 1] = _dyOfLastLayer;
            int miniBatchSize = _dyOfLastLayer.Shape[0];

            var firstTrainableLayer = _network.FirstTrainableLayer();
            if (firstTrainableLayer == null)
            {
                //the network has all its weights frozen
                return;
            }
            var firstTrainableLayerIndex = firstTrainableLayer.LayerIndex;

            for (int layerIndex = _network.Layers.Count - 1; layerIndex >= firstTrainableLayerIndex; --layerIndex)
            {
                //we are in the layer at index 'layerIndex'
                //we already know 'dy' for this layer
                //we want to compute dx (& weight gradients if the layer has weights) of current layer by backward propagation
                var layer = _network.Layers[layerIndex];
                Network.StartTimer(layer.Type(), _network.BackwardPropagationTime);
                var dy = layerIndex_to_dY[layerIndex];
                Debug.Assert(dy != null);

                //we create the buffers for the 'dx' tensors of current layer
                var dxBuffer = layer.PreviousLayers.Select(prev => prev.IsInputLayer?null:GetTensorOfShape(prev.OutputShape(miniBatchSize), isMock)).ToList();
                if (!isMock)
                {
                    Debug.Assert(layer.y.SameShape(dy));
                    //computes 'dx' and weight gradients of current layer
                    layer.BackwardPropagation(dy, dxBuffer);
                }

                //we'll update/store the output gradients (dy) of all previous layers connected to the current layer
                for (int i = 0; i < layer.PreviousLayers.Count; ++i)
                {
                    var prevLayerIndex = layer.PreviousLayers[i].LayerIndex;
                    //'dx' (input gradient) of current layer is the same as 'dy' (output gradient) of previous layer
                    var prevLayerdY = dxBuffer[i];

                    if (prevLayerIndex < firstTrainableLayerIndex)
                    {
                        //there is no need to compute/keep 'dy' of layer 'prevLayerIndex'
                        //we do not need to do any back propagation for it
                        if (prevLayerdY != null)
                        {
                            AddInCache(prevLayerdY);
                        }
                        continue;
                    }

                    if (layerIndex_to_dY[prevLayerIndex] == null)
                    {
                        layerIndex_to_dY[prevLayerIndex] = prevLayerdY;
                    }
                    else
                    {
                        if (!isMock)
                        {
                            //we'll add the content of 'prevLayerdY' to an existing gradient
                            //it means that the output of 'prevLayer' is consumed by several layers
                            layerIndex_to_dY[prevLayerIndex].Update_Adding_Alpha_X(1, prevLayerdY);
                        }
                        //we can free (discard) the content of prevLayer dY : it has already been added to an existing tensor
                        AddInCache(prevLayerdY);
                    }
                }

                if (layerIndex != _network.Layers.Count - 1)
                {
                    //we put back 'dy' in the cache because it is not used anymore
                    AddInCache(dy);
                }
                layerIndex_to_dY[layerIndex] = null;

                Network.StopTimer(layer.Type(), _network.BackwardPropagationTime);
            }
        }
        public void Dispose()
        {
            _availableTensorOrderedByCount.ForEach(t => t?.Dispose());
            _availableTensorOrderedByCount.Clear();
            _dyOfLastLayer?.Dispose();
            _dyOfLastLayer = null;
        }

        private void AddInCache(Tensor t)
        {
            _availableTensorOrderedByCount.Add(t);
            _availableTensorOrderedByCount.Sort((x, y) => (int) (CapacityInBytes(x) - CapacityInBytes(y)));
        }

        private Tensor GetTensorOfShape(int[] shape, bool isMock)
        {
            var neededMemoryInBytes = NeededMemoryInBytes(shape);
            for (var i = 0; i < _availableTensorOrderedByCount.Count; i++)
            {
                var availableTensor = _availableTensorOrderedByCount[i];
                if (CapacityInBytes(availableTensor) >= neededMemoryInBytes)
                {
                    _availableTensorOrderedByCount.RemoveAt(i);
                    if (!isMock)
                    {
                        availableTensor.Reshape(shape);
                    }
                    return availableTensor;
                }
            }
            var newTensor = NewNotInitializedFloatTensor(shape, isMock);
            return newTensor;
        }
        private ulong NeededMemoryInBytes(int[] shape)
        {
            return (ulong)(Utils.Product(shape) * _network.Config.TypeSize);
        }
        private static ulong CapacityInBytes(Tensor t)
        {
            if (IsMock(t))
            {
                return (ulong)((CpuTensor<float>)t).Content[0];
            }
            return t.CapacityInBytes;
        }
        private static bool IsMock(Tensor t)
        {
            return t.Description == MockKey;
        }
        private Tensor NewNotInitializedFloatTensor(int[] shape, bool isMock)
        {
            if (isMock)
            {
                return new CpuTensor<float>(new[] {1}, new float[]{ NeededMemoryInBytes(shape) }, MockKey);
            }
            return _network.NewNotInitializedFloatTensor(shape, "dy");
        }
    }
}