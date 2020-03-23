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
                _dyOfLastLayer = NewNotInitializedTensor(_network.Layers.Last().OutputShape(1), true);
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
                _dyOfLastLayer = _network.NewNotInitializedTensor(shape, _dyOfLastLayer, nameof(dyOfLastLayer));
                return _dyOfLastLayer;
            }
        }
        public void BackwardPropagation()
        {
            bool isMock = IsMock(_dyOfLastLayer);
            var _layerTensor_dY = Enumerable.Repeat((Tensor)null, _network.Layers.Count).ToList();
            _layerTensor_dY[_layerTensor_dY.Count- 1] = _dyOfLastLayer;
            int miniBatchSize = _dyOfLastLayer.Shape[0];
            for (int i = _network.Layers.Count - 1; i >= 1; --i)
            {
                var currentLayer = _network.Layers[i];
                var dyCurrentLayer = _layerTensor_dY[i];
                Debug.Assert(dyCurrentLayer != null);
                var dx = currentLayer.PreviousLayers.Select(prev => prev.IsInputLayer?null:GetTensorOfShape(prev.OutputShape(miniBatchSize), isMock)).ToList();
                if (!isMock)
                {
                    Debug.Assert(currentLayer.y.SameShape(dyCurrentLayer));
                    currentLayer.BackwardPropagation(dyCurrentLayer, dx);
                }
                for (int j = 0; j < currentLayer.PreviousLayers.Count; ++j)
                {
                    var prevLayer = currentLayer.PreviousLayers[j];
                    var dyPrevLayer = dx[j];
                    if (_layerTensor_dY[prevLayer.LayerIndex] == null)
                    {
                        _layerTensor_dY[prevLayer.LayerIndex] = dyPrevLayer;
                    }
                    else
                    {
                        if (!isMock)
                        {
                            _layerTensor_dY[prevLayer.LayerIndex].Update_Adding_Alpha_X(1, dyPrevLayer);
                        }
                        AddInCache(dyPrevLayer);
                    }
                }

                //we put back 'dyCurrentLayer' in the cache because it is not used anymore
                if (i != _network.Layers.Count - 1)
                {
                    AddInCache(dyCurrentLayer);
                }
                _layerTensor_dY[i] = null;
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
            var newTensor = NewNotInitializedTensor(shape, isMock);
            return newTensor;
        }
        private ulong NeededMemoryInBytes(int[] shape)
        {
            return (ulong)(Utils.Product(shape) * _network.Config.TypeSize);
        }
        private ulong CapacityInBytes(Tensor t)
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
        private Tensor NewNotInitializedTensor(int[] shape, bool isMock)
        {
            if (isMock)
            {
                return new CpuTensor<float>(new[] {1}, new float[]{ NeededMemoryInBytes(shape) }, MockKey);
            }
            return _network.NewNotInitializedTensor(shape, "dy");
        }
    }
}