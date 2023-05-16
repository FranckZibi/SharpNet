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
        /// <param name="allXForInputLayers">for each InputLayer in the Network, the associate input of this Input Layer</param>
        /// <param name="yPredicted">if provided: the buffer where to store the prediction</param>
        /// <param name="isTraining">
        ///     true if we are training the network (the goal is to update weights)
        ///     false for inference only (we'll use existing weights to make a prediction)
        /// </param>
        /// <returns></returns>
        public void Forward([NotNull] List<Tensor> allXForInputLayers, [NotNull] Tensor yPredicted, bool isTraining)
        {
            FreeAllMemory();
            Debug.Assert(_all_allocated_Y.Count == 0);
            var stopwatchDico = isTraining ? _forwardPropagationTrainingTime : _forwardPropagationInferenceTime;
            // we ensure that we have the right number of elements in 'allXForInputLayers'
            // it should be equal to the number of Input Layer in the Network
            int inputLayerCount = _layers.Count(l => l.IsInputLayer);
            if (inputLayerCount != allXForInputLayers.Count)
            {
                throw new ArgumentException("invalid number of X input ("+ allXForInputLayers.Count+") compared to number of InputLayer ("+ inputLayerCount + ")");
            }

            //referenceCountToLayer[layerIndex] : number of layers using the output of layer 'layerIndex'
            var referenceCountToLayer = new List<int>();
            int batchSize = allXForInputLayers[0].Shape[0];

            var firstTrainableLayer = Layer.FirstTrainableLayer(_layers);
            var lastLayerIndex = _layers.Last().LayerIndex;
            int inputLayerProcessed = 0;

            for (var layerIndex = 0; layerIndex <= lastLayerIndex; layerIndex++)
            {
                var layer = _layers[layerIndex];
                Network.StartTimer(layer.LayerType(), stopwatchDico);

                var allLayerInput = layer.PreviousLayerIndexes.Select(i => _all_allocated_Y[i]).ToList();
                referenceCountToLayer.Add(layer.NextLayerIndexes.Count);

                Tensor yBuffer = null;
                if (layer.IsInputLayer)
                {
                    var X = allXForInputLayers[inputLayerProcessed++];
                    // The Height & Width in allowed to change from one batch to another.
                    // But all elements in the same batch must have exactly the same shape
                    ((InputLayer)layer).SetInputHeightAndWidth(X.Shape.Length >= 3 ? X.Shape[2] : -1, X.Shape.Length >= 4 ? X.Shape[3] : -1);
                    _all_allocated_Y.Add(X);
                    yBuffer = X;
                }
                else
                {
                    if (layerIndex == lastLayerIndex)
                    {
                        yBuffer = yPredicted;
                    }
                    else
                    {
                        yBuffer = Get_yBuffer(layer, batchSize);
                        _all_allocated_Y.Add(yBuffer);
                    }
                    layer.ForwardPropagation(allLayerInput, yBuffer, isTraining);
                }


                if (LogPropagation)
                {
                    layer.LogDebug(Environment.NewLine+ "--------------------------------------------------------------------"
                                   + Environment.NewLine + "Forward: "
                                   + layer);
                    layer.Parameters.ForEach(v=> layer.LogDebug(v.Item2 + " " + v.Item1.ToShapeAndNumpy()));
                    //layer.Parameters.ForEach(v=> layer.LogDebug(v.Item2 + ": " + v.Item1.ContentStats()));
                    layer.LogDebug("output: " + yBuffer?.ToShapeAndNumpy());
                    layer.LogDebug("output:" + yBuffer?.ContentStats());
                    if (layerIndex == lastLayerIndex)
                    {
                        layer.LogDebug(Environment.NewLine + "--------------------------------------------------------------------");
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
                        if (!_layers[previousLayerIndex].IsInputLayer)
                        {
                            //we can not collect the input 'x' tensor
                            _memoryPool.FreeFloatTensor(_all_allocated_Y, previousLayerIndex);
                        }
                        _all_allocated_Y[previousLayerIndex] = null;
                    }
                }
                Network.StopTimer(layer.LayerType(), stopwatchDico);
            }
            Debug.Assert(referenceCountToLayer.Max() == 0);
            Debug.Assert(_all_allocated_Y.Count  == (_layers.Count-1)); //the output of the last layer (yPredicted) should not be put on '_all_allocated_Y'
        }

        private Tensor Get_yBuffer(Layer layer, int batchSize)
        {
            var outputShape = layer.OutputShape(batchSize);
            return _memoryPool.GetFloatTensor(outputShape);
        }

        private Tensor Get_dxBuffer(Layer prev, int batchSize)
        {
            var outputShape = prev.OutputShape(batchSize);
            return _memoryPool.GetFloatTensor(outputShape);
        }
       
        public void Backward([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted, EvaluationMetricEnum evaluationMetric)
        {
            //Debug.Assert(yExpected.SameShape(yPredicted));
            var firstTrainableLayer = Layer.FirstTrainableLayer(_layers);
            var lastLayerIndex = _layers.Last().LayerIndex;

            var dyPredicted = _memoryPool.GetFloatTensor(yPredicted.Shape);

            switch (evaluationMetric)
            {
                case EvaluationMetricEnum.BinaryCrossentropy:
                    Debug.Assert(_layers.Last().IsSigmoidActivationLayer());
                    //we compute: _dyPredicted = (1.0/categoryCount) * (yPredicted - yExpected)
                    yPredicted.CopyTo(dyPredicted);
                    var categoryCount = yPredicted.Shape[1];
                    var multiplier = 1f / (categoryCount);
                    dyPredicted.AddTensor(-multiplier, yExpected, multiplier);
                    break;
                case EvaluationMetricEnum.CategoricalCrossentropy:
                    Debug.Assert(_layers.Last().IsSoftmaxActivationLayer());
                    //we compute: _dyPredicted = (yPredicted - yExpected)
                    yPredicted.CopyTo(dyPredicted);
                    dyPredicted.AddTensor(-1, yExpected, 1);
                    break;
                case EvaluationMetricEnum.SparseCategoricalCrossentropy:
                    dyPredicted.SparseCategoricalCrossentropyGradient(yExpected, yPredicted);
                    break;
                case EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy:
                    dyPredicted.CategoricalCrossentropyWithHierarchyGradient(yExpected, yPredicted);
                    break;
                case EvaluationMetricEnum.Huber:
                    const float huberDelta = 1.0f;
                    dyPredicted.HuberGradient(yExpected, yPredicted, huberDelta);
                    break;
                case EvaluationMetricEnum.Mse:
                    dyPredicted.MseGradient(yExpected, yPredicted);
                    break;
                case EvaluationMetricEnum.Mae:
                    dyPredicted.MaeGradient(yExpected, yPredicted);
                    break;
                default:
                    throw new Exception("Invalid loss function " + evaluationMetric);
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
                var dy = all_dY[layerIndex];
                if (layer.IsInputLayer)
                {
                    Debug.Assert(dy == null);
                    continue;
                }
                Network.StartTimer(layer.LayerType(), _backwardPropagationTime);
                Debug.Assert(dy != null);
                var y = (layerIndex == lastLayerIndex)? yPredicted : _all_allocated_Y[layerIndex];

                //we allocate the buffers for the 'dx' tensors of current layer
                var dxBuffer = layer.PreviousLayers.Select(prev => prev.IsInputLayer ? null : Get_dxBuffer(prev, miniBatchSize)).ToList();
                var allX = layer.PreviousLayerIndexes.Select(i => _all_allocated_Y[i]).ToList();
                if (!layer.InputNeededForBackwardPropagation)
                {
                    allX.Clear();
                }
                if (!layer.OutputNeededForBackwardPropagation)
                {
                    y = null;
                }

                //computes 'dx' and weight gradients of current layer
                layer.BackwardPropagation(allX, y, dy, dxBuffer);

                if (LogPropagation)
                {
                    layer.LogDebug("backward: "+layer);
                    if (layer.WeightGradients != null)
                    {
                        layer.LogDebug("dW: " + layer.WeightGradients.ToShapeAndNumpy());
                        //layer.LogDebug("dW: " + layer.WeightGradients.ContentStats());
                    }
                    if (layer.BiasGradients != null)
                    {
                        layer.LogDebug("dB: " + layer.BiasGradients.ToShapeAndNumpy());
                        //layer.LogDebug("dB: " + layer.BiasGradients.ContentStats());
                    }
                    for (var index = 0; index < dxBuffer.Count; index++)
                    {
                        //layer.LogDebug("dx["+index+ "]: " + dxBuffer[index]?.ToShapeAndNumpy());
                        layer.LogDebug("dx["+index+ "]: " + dxBuffer[index]?.ContentStats());
                    }
                    layer.LogDebug("");
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
                        //we'll add the content of 'prevLayerdY' to an existing gradient
                        //it means that the output of 'prevLayer' is consumed by several layers
                        all_dY[prevLayerIndex].Update_Adding_Alpha_X(1, prevLayerdY);
                        //we can free (discard) the content of prevLayer dY : it has already been added to an existing tensor
                        _memoryPool.FreeFloatTensor(ref prevLayerdY);
                    }
                }

                //we put back 'dy' in the cache because it is not used anymore
                _memoryPool.FreeFloatTensor(all_dY, layerIndex);

                //we put back 'y' in the cache because it is not used anymore
                if ((layerIndex != lastLayerIndex)&& !layer.IsInputLayer)
                {
                    _memoryPool.FreeFloatTensor(_all_allocated_Y, layerIndex);
                }
                if (layerIndex < _all_allocated_Y.Count)
                {
                    _all_allocated_Y[layerIndex] = null;
                }
                Network.StopTimer(layer.LayerType(), _backwardPropagationTime);
            }
            FreeAllMemory();
        }

       
        public void UpdateWeights(int batchSize, double learningRate, double maxLearningRate)
        {
            var firstTrainableLayer = Layer.FirstTrainableLayer(_layers);
            if (firstTrainableLayer == null)
            {
                return;
            }
            for (var index = firstTrainableLayer.LayerIndex; index < _layers.Count; index++)
            {
                var layer = _layers[index];
                Network.StartTimer(layer.LayerType(), _updateWeightsTime);
                layer.UpdateWeights(batchSize, learningRate, maxLearningRate);
                Network.StopTimer(layer.LayerType(), _updateWeightsTime);
            }
        }

        private void FreeAllMemory()
        {
            for(int i=0;i< _all_allocated_Y.Count;++i)
            {
                if (_all_allocated_Y[i] == null)
                {
                    continue;
                }
                if (_layers.Count>i && _layers[i].IsInputLayer)
                {
                    continue;
                }
                _memoryPool.FreeFloatTensor(_all_allocated_Y[i]);
            }
            _all_allocated_Y.Clear();
        }

        public void Dispose()
        {
            FreeAllMemory();
        }
    }
}
