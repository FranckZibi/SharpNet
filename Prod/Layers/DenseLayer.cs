﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    /// <summary>
    /// output shape :
    ///     (batchSize, n_x)
    /// </summary>
    public sealed class DenseLayer : Layer
    {
        #region Private fields
        #region trainable parameters
        /// <summary>
        /// shape: (prevLayer.n_x, n_x)
        /// </summary>
        [NotNull] private Tensor _weights;
        /// <summary>
        /// shape: (1, n_x)
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
        public int CategoryCount { get; }
        #endregion

        public DenseLayer(int categoryCount, double lambdaL2Regularization, Network network, string layerName) : base(network, layerName)
        {
            CategoryCount = categoryCount;
            LambdaL2Regularization = lambdaL2Regularization;

            //trainable params
            _weights = GetFloatTensor(new[] { PrevLayer.n_x, CategoryCount });
            _bias = GetFloatTensor(new[] {1, CategoryCount });
            Debug.Assert(_bias != null);

            _weightGradients = GetFloatTensor(_weights.Shape);
            _biasGradients = (_bias != null) ? GetFloatTensor(_bias.Shape) : null;

            _optimizer = Network.GetOptimizer(_weights.Shape, _bias?.Shape);
            ResetWeights(false);
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            //We compute y = x*Weights+B
            y.Dot(x, _weights);
            _bias?.BroadcastAddVectorToOutput(y);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);
            var x = allX[0];
            int batchSize = dy.Shape[0];

            //we compute dW
            var multiplier = 1f / batchSize;
            if (Network.Config.TensorFlowCompatibilityMode)
            {
                multiplier = 1f; //used only for tests and parallel run
            }
            _weightGradients.Dot(x, true, dy, false, multiplier, 0);

            //L2 regularization on dW
            if (UseL2Regularization)
            {
                var alpha = 2 * batchSize * (float)LambdaL2Regularization;
                _weightGradients.Update_Adding_Alpha_X(alpha, _weights);
            }

            if (UseBias)
            {
                Debug.Assert(_bias != null);
                dy.Compute_BiasGradient_from_dy(_biasGradients);
            }

            //no need to compute dx (= PrevLayer.dy) if previous Layer it is the input layer
            if (PrevLayer.IsInputLayer)
            {
                return;
            }

            // we compute dx = dy * Weights.T
            dx[0].Dot(dy, false, _weights, true, 1, 0);
        }
        #endregion

        #region parameters and gradients
        public override Tensor Weights => _weights;
        public override Tensor WeightGradients => _weightGradients;
        public override Tensor Bias => _bias;
        public override Tensor BiasGradients => _biasGradients;
        protected override Optimizer Optimizer => _optimizer;
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            //trainable params
            _weights.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, Math.Sqrt(2.0 / PrevLayer.n_x) /*stdDev*/);
            _bias?.ZeroMemory();

            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }
        public override void SetParameters(List<Tensor> newParameters)
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
        public override void SetGradients(List<Tensor> newGradients)
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

        #region serialization
        public override string Serialize()
        {
            return RootSerializer() // 'RootSerializer()' will also serialize layer trainable params
                .Add(nameof(CategoryCount), CategoryCount)
                .Add(nameof(LambdaL2Regularization), LambdaL2Regularization)
                .Add(_optimizer.Serialize())
                .ToString();
        }
        public DenseLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            CategoryCount = (int)serialized[nameof(CategoryCount)];
            LambdaL2Regularization = (double)serialized[nameof(LambdaL2Regularization)];

            //trainable params
            var useBias = serialized.ContainsKey(nameof(_bias));
            _weights = (Tensor)serialized[nameof(_weights)];
            _bias = useBias ? (Tensor)serialized[nameof(_bias)] : null;

            //gradients
            _weightGradients = GetFloatTensor(_weights.Shape);
            _biasGradients = (_bias != null) ? GetFloatTensor(_bias.Shape) : null;

            _optimizer = Optimizer.ValueOf(network.Config, serialized);
        }
        #endregion

        public override void AddToOtherNetwork(Network otherNetwork)
        {
            otherNetwork.Layers.Add(new DenseLayer(CategoryCount, LambdaL2Regularization, otherNetwork, LayerName));
        }

        public override int[] OutputShape(int batchSize)
        {
            return new[] { batchSize, CategoryCount };
        }
        public override int DisableBias()
        {
            int nbDisabledWeights = (_bias?.Count ?? 0);
            FreeFloatTensor(ref _bias);
            FreeFloatTensor(ref _biasGradients);
            return nbDisabledWeights;
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
        public override void LoadFromH5Dataset(Dictionary<string, Tensor> h5FileDataset, NetworkConfig.CompatibilityModeEnum originFramework)
        {
            //var cpuTensor = (CpuTensor<float>)h5FileDataset[weightDatasetPath];
            //var reshapedCpuTensor = cpuTensor.WithNewShape(new[] { cpuTensor.Shape[0], cpuTensor.Shape[1], 1, 1 });
            h5FileDataset[DatasetNameToDatasetPath("kernel:0")].CopyTo(_weights);

            //var biasCpuTensor = (CpuTensor<float>)h5FileDataset[biasDatasetPath];
            //var reshapedBiasCpuTensor = biasCpuTensor.WithNewShape(new[] {1, biasCpuTensor.Shape[0]});
            h5FileDataset[DatasetNameToDatasetPath("bias:0")].CopyTo(_bias);
        }
   
        private bool UseL2Regularization => LambdaL2Regularization > 0.0;
        private bool UseBias => _bias != null;
    }
}
