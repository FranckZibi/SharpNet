using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.Layers;

namespace SharpNet.Networks
{
    public class PropagationManager : IDisposable
    {
        #region private fields
        private readonly List<Layer> _layers;
        [NotNull] private readonly TensorMemoryPool _memoryPool;
        [NotNull] private readonly IDictionary<string, Stopwatch> _forwardPropagationTrainingTime;
        [NotNull] private readonly IDictionary<string, Stopwatch> _forwardPropagationInferenceTime;
        [NotNull] private readonly IDictionary<string, Stopwatch> _backwardPropagationTime;
        [NotNull] private readonly IDictionary<string, Stopwatch> _updateWeightsTime;

        // output of each layer, without the last one (prediction)
        private readonly List<Tensor> _all_allocated_Y = new List<Tensor>();
        #endregion

        public PropagationManager(List<Layer> layers, [NotNull] TensorMemoryPool memoryPool, [NotNull] IDictionary<string, Stopwatch> forwardPropagationTrainingTime, [NotNull] IDictionary<string, Stopwatch> forwardPropagationInferenceTime, [NotNull] IDictionary<string, Stopwatch> backwardPropagationTime, [NotNull] IDictionary<string, Stopwatch> updateWeightsTime)
        {
            _layers = layers;
            _memoryPool = memoryPool;
            _forwardPropagationTrainingTime = forwardPropagationTrainingTime;
            _forwardPropagationInferenceTime = forwardPropagationInferenceTime;
            _backwardPropagationTime = backwardPropagationTime;
            _updateWeightsTime = updateWeightsTime;
        }

        /// <summary>
        /// when set to true, will log all forward and backward propagation
        /// </summary>
        // ReSharper disable once MemberCanBePrivate.Global
        public bool LogPropagation { get; set; }

        /// <summary>
        /// = ForwardPropagation
        /// </summary>
        /// <param name="X"></param>
        /// <param name="yPredicted">if provided: the buffer where to store the prediction</param>
        /// <param name="isTraining">
        ///     true if we are training the network (the goal is to update weights)
        ///     false for inference only (we'll use existing weights to make a prediction)
        /// </param>
        /// <returns></returns>
        public void Forward([NotNull] Tensor X, [NotNull] Tensor yPredicted, bool isTraining)
        {
            FreeAllMemory();
            Debug.Assert(_all_allocated_Y.Count == 0);
            var stopwatchDico = isTraining ? _forwardPropagationTrainingTime : _forwardPropagationInferenceTime;
            ((InputLayer)_layers[0]).SetInputShape(X.Shape);
            //referenceCountToLayer[layerIndex] : number of layers using the output of layer 'layerIndex'
            var referenceCountToLayer = new List<int>();
            int batchSize = X.Shape[0];
            _all_allocated_Y.Add(X); //input layer (layerIndex = 0) as output 'X'

            var firstTrainableLayer = Layer.FirstTrainableLayer(_layers);
            var lastLayerIndex = _layers.Last().LayerIndex;
            referenceCountToLayer.Add(_layers[0].NextLayerIndexes.Count);

            for (var layerIndex = 1; layerIndex <= lastLayerIndex; layerIndex++)
            {
                var layer = _layers[layerIndex];
                Network.StartTimer(layer.Type(), stopwatchDico);
                Tensor yBuffer;
                if (layerIndex == lastLayerIndex)
                {
                    yBuffer = yPredicted;
                }
                else
                {
                    yBuffer = Get_yBuffer(layer, batchSize);
                    _all_allocated_Y.Add(yBuffer);
                }
                referenceCountToLayer.Add(layer.NextLayerIndexes.Count);
                var allX = layer.PreviousLayerIndexes.Select(i => _all_allocated_Y[i]).ToList();
                if (!_memoryPool.IsMock)
                {
                    layer.ForwardPropagation(allX, yBuffer, isTraining);

                    if (LogPropagation)
                    {
                        layer.Log(layer.ToString());
                        layer.Parameters.ForEach(v=> layer.LogDebug(v.Item2 + ": " + Environment.NewLine + v.Item1.ToNumpy()));
                        layer.LogDebug("output:" + Environment.NewLine + yBuffer.ToNumpy());
                        layer.LogDebug("");
                    }
                }

                //we collect any output layers that are not needed anymore
                foreach (var previousLayerIndex in layer.PreviousLayerIndexes)
                {
                    --referenceCountToLayer[previousLayerIndex];
                    Debug.Assert(referenceCountToLayer[previousLayerIndex] >= 0);
                    Debug.Assert(_all_allocated_Y[previousLayerIndex] != null);

                    if (  //if the output of layer 'previousLayerIndex' is not used anymore
                          referenceCountToLayer[previousLayerIndex] == 0
                        //and if we can collect the output tensor of layer 'previousLayerIndex' because it will not be used by backward propagation
                        && !_layers[previousLayerIndex].LayerOutputShouldBeKeptForBackwardPropagation(isTraining, firstTrainableLayer) )
                    {
                        if (previousLayerIndex != 0)
                        {
                            //we can not collect the input 'x' tensor
                            _memoryPool.FreeFloatTensor(_all_allocated_Y, previousLayerIndex);
                        }
                        _all_allocated_Y[previousLayerIndex] = null;
                    }
                }
                Network.StopTimer(layer.Type(), stopwatchDico);
            }
            Debug.Assert(referenceCountToLayer.Max() == 0);
            Debug.Assert(_all_allocated_Y.Count  == (_layers.Count-1)); //the output of the last layer (yPredicted) should not be put on '_all_allocated_Y'
        }

        private Tensor Get_yBuffer(Layer layer, int batchSize)
        {
            var outputShape = layer.OutputShape(batchSize);
            if (!_memoryPool.IsMock || 0 == layer.ExtraElementCountForForwardPropagation(batchSize))
            {
                return _memoryPool.GetFloatTensor(outputShape);
            }
            return _memoryPool.GetFloatTensor(ReshapeWithExtraElementCount(outputShape, layer.ExtraElementCountForForwardPropagation(batchSize)));
        }

        private Tensor Get_dxBuffer(Layer prev, int batchSize)
        {
            var outputShape = prev.OutputShape(batchSize);
            if (!_memoryPool.IsMock || 0 == prev.ExtraElementCountForBackwardPropagation(batchSize))
            {
                return _memoryPool.GetFloatTensor(outputShape);
            }
            return _memoryPool.GetFloatTensor(ReshapeWithExtraElementCount(outputShape, prev.ExtraElementCountForBackwardPropagation(batchSize)));
        }

        private static int[] ReshapeWithExtraElementCount(int[] initialShape, int extraElementCount)
        {
            var batchSize = initialShape[0];
            var currentElementCount = Utils.Product(initialShape);
            return new[] { batchSize, (currentElementCount + extraElementCount) / batchSize, 1, 1 };
        }

        public void Backward([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted)
        {
            Debug.Assert(yExpected != null);
            Debug.Assert(yExpected.SameShape(yPredicted));
            Tensor dyPredicted;
            var firstTrainableLayer = Layer.FirstTrainableLayer(_layers);
            var lastLayerIndex = _layers.Last().LayerIndex;

            if (_memoryPool.IsMock)
            {
                dyPredicted = _memoryPool.GetFloatTensor(_layers.Last().OutputShape(1));
            }
            else
            {
                //we compute: _dyPredicted = (1.0 / categoryCount)*(yPredicted - yExpected)
                dyPredicted = _memoryPool.GetFloatTensor(yExpected.Shape);
                yPredicted.CopyTo(dyPredicted);
                var categoryCount = yPredicted.Shape[1];
                var multiplier = _layers.Last().IsSigmoidActivationLayer() ? (1f / categoryCount) : 1f;
                dyPredicted.AddTensor(-multiplier, yExpected, multiplier);
            }

            var all_dY = new Tensor[_layers.Count];
            all_dY[lastLayerIndex] = dyPredicted;
            int miniBatchSize = dyPredicted.Shape[0];

            if (firstTrainableLayer == null)
            {
                //the network has all its weights frozen
                FreeAllMemory();
                return;
            }
            var firstTrainableLayerIndex = firstTrainableLayer.LayerIndex;

            for (int layerIndex = lastLayerIndex; layerIndex >= firstTrainableLayerIndex; --layerIndex)
            {
                //we are in the layer at index 'layerIndex'
                //we already know 'dy' for this layer ( = all_dY[layerIndex])
                //we want to compute dx (& weight gradients if the layer has weights) of current layer by backward propagation
                var layer = _layers[layerIndex];
                Network.StartTimer(layer.Type(), _backwardPropagationTime);
                var dy = all_dY[layerIndex];
                Debug.Assert(dy != null);
                var y = (layerIndex == lastLayerIndex)? yPredicted : _all_allocated_Y[layerIndex];
                Debug.Assert(y != null);

                //we allocate the buffers for the 'dx' tensors of current layer
                var dxBuffer = layer.PreviousLayers.Select(prev => prev.IsInputLayer ? null : Get_dxBuffer(prev, miniBatchSize)).ToList();
                var allX = layer.PreviousLayerIndexes.Select(i => _all_allocated_Y[i]).ToList();
                if (!_memoryPool.IsMock)
                {
                    //computes 'dx' and weight gradients of current layer
                    layer.BackwardPropagation(allX, y, dy, dxBuffer);

                    if (LogPropagation)
                    {
                        layer.Log(layer.ToString());
                        if (layer.WeightGradients != null)
                        {
                            layer.Log("dW: " + Environment.NewLine + layer.WeightGradients.ToNumpy());
                        }
                        if (layer.BiasGradients != null)
                        {
                            layer.Log("dB: " + Environment.NewLine + layer.BiasGradients.ToNumpy());
                        }
                        for (var index = 0; index < dxBuffer.Count; index++)
                        {
                            layer.Log("dx["+index+ "]: " + Environment.NewLine + dxBuffer[index].ToNumpy());
                        }
                        layer.Log("");
                    }









                }

                //we'll update/store the output gradients (dy) of all previous layers connected to the current layer
                for (int i = 0; i < layer.PreviousLayerIndexes.Count; ++i)
                {
                    var prevLayerIndex = layer.PreviousLayerIndexes[i];
                    //'dx' (input gradient) of current layer is the same as 'dy' (output gradient) of previous layer
                    var prevLayerdY = dxBuffer[i];

                    if (prevLayerIndex < firstTrainableLayerIndex)
                    {
                        //there is no need to compute/keep 'dy' of layer 'prevLayerIndex'
                        //we do not need to do any back propagation for it
                        _memoryPool.FreeFloatTensor(ref prevLayerdY);
                        continue;
                    }

                    if (all_dY[prevLayerIndex] == null)
                    {
                        all_dY[prevLayerIndex] = prevLayerdY;
                    }
                    else
                    {
                        if (!_memoryPool.IsMock)
                        {
                            //we'll add the content of 'prevLayerdY' to an existing gradient
                            //it means that the output of 'prevLayer' is consumed by several layers
                            all_dY[prevLayerIndex].Update_Adding_Alpha_X(1, prevLayerdY);
                        }
                        //we can free (discard) the content of prevLayer dY : it has already been added to an existing tensor
                        _memoryPool.FreeFloatTensor(ref prevLayerdY);
                    }
                }

                //we put back 'dy' in the cache because it is not used anymore
                _memoryPool.FreeFloatTensor(all_dY, layerIndex);

                //we put back 'y' in the cache because it is not used anymore
                if ((layerIndex != lastLayerIndex)&& (layerIndex != 0))
                {
                    _memoryPool.FreeFloatTensor(_all_allocated_Y, layerIndex);
                }
                if (layerIndex < _all_allocated_Y.Count)
                {
                    _all_allocated_Y[layerIndex] = null;
                }
                Network.StopTimer(layer.Type(), _backwardPropagationTime);
            }
            FreeAllMemory();
        }

       
        public void UpdateWeights(int batchSize, double learningRate)
        {
            var firstTrainableLayer = Layer.FirstTrainableLayer(_layers);
            if (firstTrainableLayer == null)
            {
                return;
            }
            for (var index = firstTrainableLayer.LayerIndex; index < _layers.Count; index++)
            {
                var layer = _layers[index];
                Network.StartTimer(layer.Type(), _updateWeightsTime);
                layer.UpdateWeights(batchSize, learningRate);
                Network.StopTimer(layer.Type(), _updateWeightsTime);
            }
        }

        private void FreeAllMemory()
        {
            if (_all_allocated_Y.Count >= 1)
            {
                _all_allocated_Y[0] = null;
                _all_allocated_Y.ForEach(_memoryPool.FreeFloatTensor);
            }
            _all_allocated_Y.Clear();
        }

        public void Dispose()
        {
            FreeAllMemory();
        }
    }
}
