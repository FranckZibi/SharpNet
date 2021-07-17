using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    /// <summary>
    /// input shape :
    ///     (batchSize, ..., n_x)
    /// output shape :
    ///     (batchSize, ..., units)                 if flattenInputTensorOnLastDimension == true
    ///     (batchSize, units)                      if flattenInputTensorOnLastDimension == false
    /// </summary>
    public sealed class DenseLayer : Layer
    {
        #region Private fields
        #region trainable parameters
        /// <summary>
        /// shape: 
        ///    (prevLayerOutputShape[last], units)      if _flattenInputTensorOnLastDimension == true
        ///    (PrevLayer.n_x, units)                   if _flattenInputTensorOnLastDimension == false
        /// </summary>
        [NotNull] private Tensor _weights;
        /// <summary>
        /// shape: (1, Units)
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
        /// regularization hyper parameter. 0 if no L2 regularization
        /// </summary>
        public double LambdaL2Regularization { get; }
        /// <summary>
        /// dimensionality of the output space
        /// </summary>
        public int Units { get; }
        /// <summary>
        /// if true
        ///     we'll flatten the input tensor x keeping the last dimension intact:     (a,b,c,d) => (a*b*c*, d)
        ///     the weight matrix will be of shape (d, units)
        /// else
        ///     we'll assume that a Flatten Layer was just before the current 'this' Layer
        ///     we'll flatten the input tensor 'x' keeping the fist dimension intact:   (a,b,c,d) => (a, b*c*d)
        ///     the weight matrix will be of shape (b*c*d, units)
        /// </summary>
        private readonly bool _flattenInputTensorOnLastDimension;
        #endregion

        #region constructor
        public DenseLayer(int units, double lambdaL2Regularization, bool flattenInputTensorOnLastDimension, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            _flattenInputTensorOnLastDimension = flattenInputTensorOnLastDimension;
            Units = units;
            LambdaL2Regularization = lambdaL2Regularization;
            Trainable = trainable;

            //trainable params
            _weights = GetFloatTensor(WeightShape);
            _bias = GetFloatTensor(new[] {1, Units });
            Debug.Assert(_bias != null);

            _weightGradients = GetFloatTensor(_weights.Shape);
            _biasGradients = (_bias != null) ? GetFloatTensor(_bias.Shape) : null;

            _optimizer = GetOptimizer(_weights.Shape, _bias?.Shape);
            ResetParameters(false);
        }
        #endregion

        private int[] WeightShape
        {
            get
            {
                if (_flattenInputTensorOnLastDimension)
                {
                    return new[] { PrevLayer.OutputShape(1).Last(), Units };
                }
                return new[] { PrevLayer.n_x, Units };
            }
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var xAs2DMatrix = As2DMatrixForDotProduct(allX[0]);
            var yAs2DMatrix = As2DMatrixForDotProduct(y);
            //We compute y = x*Weights+B
            yAs2DMatrix.Dot(xAs2DMatrix, _weights);
            _bias?.BroadcastAddVectorToOutput(yAs2DMatrix);
        }
       
        public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(dx.Count == 1);
            int batchSize = dy.Shape[0];

            var xAs2DMatrix = As2DMatrixForDotProduct(allX[0]);
            var dyAs2DMatrix = As2DMatrixForDotProduct(dy);

            //we compute dW
            var multiplier = 1f / batchSize;
            if (Config.TensorFlowCompatibilityMode)
            {
                multiplier = 1f; //used only for tests and parallel run
            }
            _weightGradients.Dot(xAs2DMatrix, true, dyAs2DMatrix, false, multiplier, 0);

            //L2 regularization on dW
            if (UseL2Regularization)
            {
                var alpha = 2 * batchSize * (float)LambdaL2Regularization;
                _weightGradients.Update_Adding_Alpha_X(alpha, _weights);
            }

            if (UseBias)
            {
                Debug.Assert(_bias != null);
                dyAs2DMatrix.Compute_BiasGradient_from_dy(_biasGradients);
            }

            //no need to compute dx (= PrevLayer.dy) if previous Layer it is the input layer
            if (PrevLayer.IsInputLayer)
            {
                return;
            }

            // we compute dx = dy * Weights.T
            dx[0].Dot(dyAs2DMatrix, false, _weights, true, 1, 0);
        }

        /// <summary>
        /// When x is tensor with >=3 dimension         (ex:  (a, b, c, d))
        ///     if _flattenInputTensorOnLastDimension == true
        ///             we'll change its shape to a 2D Matrix       (ex:  (a*b*c, d) )
        ///             so that the last dimension of the matrix (ex: d) is preserved
        ///     else (_flattenInputTensorOnLastDimension == false)
        ///             we'll change its shape to a 2D Matrix       (ex:  (a, b*c*d) )
        ///             so that the first dimension of the matrix (ex: a) is preserved
        /// </summary>
        /// <param name="x"></param>
        /// <returns>A 2D Matrix</returns>
        private Tensor As2DMatrixForDotProduct(Tensor x)
        {
            if (x.Shape.Length <= 2)
            {
                return x;
            }
            var xTargetShape = new int[2];
            if (_flattenInputTensorOnLastDimension)
            {
                xTargetShape[0] = x.Count / x.Shape.Last();
                xTargetShape[1] = x.Shape.Last();
            }
            else
            {
                xTargetShape[0] = x.Shape[0];
                xTargetShape[1] = x.Count / x.Shape[0];
            }

            return x.WithNewShape(xTargetShape);
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
            _weights.NormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / PrevLayer.n_x) /*stdDev*/);
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

        public override IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(NetworkConfig.CompatibilityModeEnum originFramework)
        {
            var result = new Dictionary<string, CpuTensor<float>>();
            result.Add(WeightDatasetPath, _weights.ToCpuFloat());
            if (UseBias)
            {
                // ReSharper disable once PossibleNullReferenceException
                result.Add(BiasDatasetPath, _bias.ToCpuFloat());
                if (originFramework == NetworkConfig.CompatibilityModeEnum.TensorFlow1 || originFramework == NetworkConfig.CompatibilityModeEnum.TensorFlow2)
                {
                    // ReSharper disable once PossibleNullReferenceException
                    Debug.Assert(_bias.Count == _bias.Shape[1]);
                    var tensorFlowShape = new[] { _bias.Shape[1] };
                    result[BiasDatasetPath].Reshape(tensorFlowShape);
                }
            }
            return result;
        }

        private string WeightDatasetPath => DatasetNameToDatasetPath("kernel:0");
        private string BiasDatasetPath => DatasetNameToDatasetPath("bias:0");
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(Units), Units)
                .Add(nameof(LambdaL2Regularization), LambdaL2Regularization)
                .Add(nameof(_flattenInputTensorOnLastDimension), _flattenInputTensorOnLastDimension)
                .ToString();
        }
        public static DenseLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var units = serialized.ContainsKey(nameof(Units)) ? (int)serialized[nameof(Units)] : (int)serialized["CategoryCount"];
            var flattenInputTensorOnLastDimension = serialized.ContainsKey(nameof(_flattenInputTensorOnLastDimension)) && (bool)serialized[nameof(_flattenInputTensorOnLastDimension)];
            return new DenseLayer(
                units,
                (double)serialized[nameof(LambdaL2Regularization)],
                flattenInputTensorOnLastDimension,
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            if (_flattenInputTensorOnLastDimension)
            {
                var outputShape = (int[]) PrevLayer.OutputShape(batchSize).Clone();
                outputShape[^1] = Units;
                return outputShape;
            }
            return new[] { batchSize, Units };
        }
        public override string ToString()
        {
            var result = LayerName+": "+ShapeChangeDescription();
            if (UseL2Regularization)
            {
                result += " with L2Regularization[lambdaValue=" + LambdaL2Regularization + "]";
            }
            result += " " + _weights+ " " + _bias + " (" +TotalParams+" neurons)";
            return result;
        }

        private bool UseL2Regularization => LambdaL2Regularization > 0.0 && Config.OptimizerType != Optimizer.OptimizationEnum.AdamW;
        private bool UseBias => _bias != null;
    }
}
