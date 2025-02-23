using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;
using static SharpNet.Networks.NetworkSample;

namespace SharpNet.Layers
{
    /// <summary>
    /// input shape :
    ///     (batchSize, a, b, ... y, z)
    /// output shape :
    ///     (batchSize, a, b, ... y, out_features)         if flattenInputTensorOnLastDimension == true
    ///     (batchSize, out_features)                      if flattenInputTensorOnLastDimension == false
    /// </summary>

    public sealed class Linear : Layer
    {
        #region Private fields
        #region trainable parameters
        /// <summary>
        /// if input shape is:
        ///     (batchSize, a, b, ... y, z)
        /// weights shape will be : 
        ///    (out_features, z)                           if _flattenInputTensorOnLastDimension == true
        ///    (out_features, a*b*...*y*z)                 if _flattenInputTensorOnLastDimension == false
        /// </summary>
        [NotNull] private Tensor _weights;
        /// <summary>
        /// shape: (1, out_features)
        /// Can be null if bias has been disabled
        /// </summary>
        [CanBeNull] private Tensor _bias;
        #endregion
        #region gradients
        /// <summary>
        /// same shape as 'Weights'
        /// </summary>
        [NotNull] private Tensor _weightGradients;
        /// <summary>
        /// same shape as 'Bias'
        /// Can be null if bias has been disabled
        /// </summary>
        [CanBeNull] private Tensor _biasGradients;
        /// <summary>
        /// Adam or SGD optimizer or Vanilla SGD
        /// </summary>
        #endregion
        [NotNull] private readonly Optimizer _optimizer;
        #endregion
        #region public fields and properties
        /// <summary>
        /// dimensionality of the output space
        /// </summary>
        public int out_features { get; }

        /// <summary>
        /// input shape :
        ///     (batchSize, a, b, ... y, z)
        /// if true
        ///     we'll flatten the input tensor x keeping the last dimension intact:     (a,b,c,d) => (a*b*c*, d)
        ///     the weight matrix will be of shape (d, out_features)
        ///     the output shape will be: (batchSize, a, b, ... y, out_features)
        ///     this is the only supported mode in PyTorch
        /// else
        ///     we'll assume that a Flatten Layer was just before the current 'this' Layer
        ///     we'll flatten the input tensor 'x' keeping the fist dimension intact:   (batchSize, a,b,c,d) => (batchSize, a, b*c*d)
        ///     the weight matrix will be of shape (b*c*d, out_features)
        ///     the output shape will be: (batchSize, out_features)
        /// </summary>
        private readonly bool _flattenInputTensorOnLastDimension;
        #endregion

        #region constructor
        public Linear(int out_features, bool bias, bool flattenInputTensorOnLastDimension, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            _flattenInputTensorOnLastDimension = flattenInputTensorOnLastDimension;
            this.out_features = out_features;
            Trainable = trainable;

            //trainable params
            _weights = GetFloatTensor(WeightShape);
            _bias = bias?GetFloatTensor(new[] {this.out_features }):null;
            
            _weightGradients = GetFloatTensor(_weights.Shape);
            _biasGradients = bias ? GetFloatTensor(_bias.Shape) : null;

            _optimizer = Sample.GetOptimizer(_weights.Shape, _bias?.Shape, MemoryPool);
            ResetParameters(false);
        }



        #endregion


        private int[] WeightShape
        {
            get
            {
                if (_flattenInputTensorOnLastDimension)
                {
                    int in_features = PrevLayer.OutputShape(1).Last();
                    return new[] { out_features, in_features };
                }
                else
                {
                    int in_features = Utils.Product(PrevLayer.OutputShape(1));
                    return new[] { out_features, in_features };
                }
            }
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            DenseForwardPropagation(y, allX[0], _weights, _bias, _flattenInputTensorOnLastDimension);
        }
        public static void DenseForwardPropagation(/* Out */ Tensor y, /* In */ Tensor x, /* In */ Tensor weights, /* In */ Tensor biasIfAny_1D, bool flattenInputTensorOnLastDimension)
        {
            var xAs2DMatrix = x.As2DTensor(flattenInputTensorOnLastDimension);
            var yAs2DMatrix = y.As2DTensor(flattenInputTensorOnLastDimension);
            //We compute y = x*Weights.t+B
            yAs2DMatrix.Dot(xAs2DMatrix, false, weights, true, 1, 0);
            biasIfAny_1D?.BroadcastAddVectorToOutput(yAs2DMatrix);
        }
       
        public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(dx.Count == 1);
            DenseBackwardPropagation(dx[0], _weightGradients, _biasGradients, allX[0], dy, _weights, _bias, Sample, Sample.weight_decay, PrevLayer.IsInputLayer, _flattenInputTensorOnLastDimension);
        }

        public static void DenseBackwardPropagation(/* Out */ Tensor dx, /* Out */ Tensor weightGradients, /* Out */ Tensor biasGradients_1D, /* In */ Tensor x, /* In */ Tensor dy, /* In */ Tensor weights, /* In */ Tensor bias_1D, NetworkSample sample, double weight_decay, bool prevLayerIsInputLayer, bool flattenInputTensorOnLastDimension)
        {
            var xAs2DMatrix = x.As2DTensor(flattenInputTensorOnLastDimension);
            var dyAs2DMatrix = dy.As2DTensor(flattenInputTensorOnLastDimension);

            Debug.Assert(xAs2DMatrix.Shape.Length == 2);
            Debug.Assert(dyAs2DMatrix.Shape.Length == 2);
            int batchSize = x.Shape[0];

            //we compute dW
            var multiplier = 1f / batchSize;
            if (sample.CompatibilityMode == CompatibilityModeEnum.TensorFlow || sample.CompatibilityMode == CompatibilityModeEnum.PyTorch)
            {
                multiplier = 1f;
            }

            weightGradients.Dot(dyAs2DMatrix, true, xAs2DMatrix, false, multiplier, 0);

            if (biasGradients_1D != null)
            {
                //Debug.Assert(_bias != null);
                //Debug.Assert(bias);
                dyAs2DMatrix.Compute_BiasGradient_from_dy(biasGradients_1D);
            }

            //weight_decay on dW (and dB for PyTorch)
            if (sample.Use_weight_decay_in_backpropagation && weight_decay>0)
            {
                var weight_decay_mulitplier = 2;
                if (sample.CompatibilityMode == CompatibilityModeEnum.PyTorch)
                {
                    weight_decay_mulitplier = 1;
                }
                var alpha = weight_decay_mulitplier * batchSize * (float)weight_decay;
                weightGradients.Update_Adding_Alpha_X(alpha, weights);
                if (sample.CompatibilityMode == CompatibilityModeEnum.PyTorch && biasGradients_1D != null)
                {
                    biasGradients_1D.Update_Adding_Alpha_X(alpha, bias_1D);
                }
            }

            //no need to compute dx (= PrevLayer.dy) if previous Layer is the input layer
            if (prevLayerIsInputLayer)
            {
                return;
            }

            // we compute dx = dy * Weights
            dx.Dot(dyAs2DMatrix, false, weights, false, 1, 0);
        }

        public override bool OutputNeededForBackwardPropagation => false;
        #endregion

        #region parameters and gradients
        public override Tensor Weights => _weights;
        public override Tensor Bias => _bias;
        public override Tensor WeightGradients => _weightGradients;
        public override Tensor BiasGradients => _biasGradients;
        protected override Optimizer Optimizer => _optimizer;
        public override List<Tuple<Tensor, string>> Parameters
        {
            get
            {
                var result = new List<Tuple<Tensor, string>>
                             {
                                 Tuple.Create(_weights, WeightDatasetPath),
                                 Tuple.Create(_bias, BiasDatasetPath)
                             };
                result.RemoveAll(t => t.Item1 == null);
                return result;
            }
        }
        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            //trainable params
            _weights.GlorotUniform(Rand);
            
            if (_optimizer.IsOrthogonal)
            {
                _weights.Orthogonal(Rand);
            }

            _bias?.ZeroMemory();
            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }

        #region Multi GPU Support
        public override void ReplaceParameters(List<Tensor> newParameters)
        {
            FreeFloatTensor(ref _weights);
            _weights = newParameters[0];
            if (_bias != null)
            {
                Debug.Assert(newParameters.Count == 2);
                FreeFloatTensor(ref _bias);
                _bias = newParameters[1];
            }
            else
            {
                Debug.Assert(newParameters.Count == 1);
            }
        }
        public override void ReplaceGradients(List<Tensor> newGradients)
        {
            FreeFloatTensor(ref _weightGradients);
            _weightGradients = newGradients[0];
            if (_biasGradients != null)
            {
                Debug.Assert(newGradients.Count == 2);
                FreeFloatTensor(ref _biasGradients);
                _biasGradients = newGradients[1];
            }
            else
            {
                Debug.Assert(newGradients.Count == 1);
            }
        }
        #endregion

        public override int DisableBias()
        {
            int nbDisabledWeights = (_bias?.Count ?? 0);
            FreeFloatTensor(ref _bias);
            FreeFloatTensor(ref _biasGradients);
            return nbDisabledWeights;
        }

        public override IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(CompatibilityModeEnum originFramework)
        {
            var result = new Dictionary<string, CpuTensor<float>>();
            result.Add(WeightDatasetPath, _weights.ToCpuFloat());
            if (UseBias)
            {
                // ReSharper disable once PossibleNullReferenceException
                result.Add(BiasDatasetPath, _bias.ToCpuFloat());
            }
            return result;
        }

        private string WeightDatasetPath => DatasetNameToDatasetPath("weight");
        private string BiasDatasetPath => DatasetNameToDatasetPath("bias");
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(out_features), out_features)
                .Add("bias", UseBias)
                .Add(nameof(_flattenInputTensorOnLastDimension), _flattenInputTensorOnLastDimension)
                .ToString();
        }
        public static Linear Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var out_features_value = serialized.ContainsKey(nameof(out_features)) ? (int)serialized[nameof(out_features)] : (int)serialized["NumClass"];
            bool bias = serialized.ContainsKey("bias") ? (bool)serialized["bias"] : true;
            var flattenInputTensorOnLastDimension = serialized.ContainsKey(nameof(_flattenInputTensorOnLastDimension)) && (bool)serialized[nameof(_flattenInputTensorOnLastDimension)];
            return new Linear(
                out_features_value,
                bias,
                flattenInputTensorOnLastDimension,
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion


        #region PyTorch support
        //see : https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
        {
            var input_shape = PreviousLayers.Count == 0 ? new[]{-1,-1} : PreviousLayers[0].OutputShape(666);
            if (_flattenInputTensorOnLastDimension || input_shape.Length == 2)
            {
                constructorLines.Add("self." + LayerName + " = torch.nn.Linear(in_features=" + input_shape[^1] + ", out_features=" + out_features + ", bias="+ Utils.ToPython(UseBias) + ")");
                UpdateForwardLines(forwardLines);
            }
            else
            {
                string flattenLayerName = "flatten_" + LayerName;
                constructorLines.Add("self." + flattenLayerName + " = torch.nn.Flatten(1)");
                var in_features = string.Join("*", input_shape.Skip(1));
                constructorLines.Add("self." + LayerName + " = torch.nn.Linear(in_features=" + in_features + ", out_features=" + out_features + ", bias=" + Utils.ToPython(UseBias) + ")");
                forwardLines.Add("y_" + flattenLayerName+" = self." + flattenLayerName + "(" + GetInputVariableName() + ")");
                forwardLines.Add(GetPyTorchOutputVariableName() + " = self." + LayerName + "(y_" + flattenLayerName + ")");
            }
            constructorLines.Add("torch.nn.init.xavier_uniform_(self." + LayerName + ".weight)");
            if (UseBias)
            {
                constructorLines.Add("torch.nn.init.zeros_(self." + LayerName + ".bias)");
            }
        }

        #endregion

        public override int[] OutputShape(int batchSize)
        {
            if (_flattenInputTensorOnLastDimension)
            {
                var outputShape = (int[]) PrevLayer.OutputShape(batchSize).Clone();
                outputShape[^1] = out_features;
                return outputShape;
            }
            return new[] { batchSize, out_features };
        }
        public override string ToString()
        {
            var result = LayerName+": "+ShapeChangeDescription();
            if (Sample.weight_decay>0)
            {
                result += " with weight_decay=" + Sample.weight_decay;
            }
            result += " " + _weights+ " " + _bias + " (" +TotalParams+" neurons)";
            return result;
        }
        private bool UseBias => _bias != null;
    }
}
