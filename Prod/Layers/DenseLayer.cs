using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    /// <summary>
    /// output shape : (batchSize, n_x)
    /// </summary>
    public sealed class DenseLayer : Layer
    {
        #region Private fields
        private readonly Tensor _weights;         // (prevLayer.n_x, n_x)
        /// <summary>
        /// (1, n_x)
        /// Can be null if bias has been disabled
        /// </summary>
        private Tensor _bias;
        /// <summary>
        /// dimensionality of the output space
        /// </summary>
        private readonly int _units;
        /// <summary>
        /// regularization hyper parameter. 0 if no L2 regularization
        /// </summary>
        public readonly double _lambdaL2Regularization;
        private bool UseBias => _bias!=null;
        private readonly Optimizer _optimizer;              //Adam or SGD optimizer or Vanilla SGF
        private Tensor _weightGradients;    // same shape as 'Weights'
        private Tensor _biasGradients;      // same shape as 'Bias'. Can be null if bias has been disabled
        #endregion

        private bool UseL2Regularization => _lambdaL2Regularization > 0.0;

        public DenseLayer(int units, double lambdaL2Regularization, Network network, string layerName) : base(network, layerName)
        {
            _units = units;
            _lambdaL2Regularization = lambdaL2Regularization;

            //trainable params
            _weights = GetNotInitializedFloatTensor(new[] { PrevLayer.n_x, _units }, nameof(_weights));
            _bias = GetNotInitializedFloatTensor(new[] {1,  _units }, nameof(_bias));
            _optimizer = Network.GetOptimizer(_weights.Shape, _bias.Shape);
            ResetWeights(false);
        }


        public override Tensor Weights => _weights;
        public override Tensor WeightGradients => _weightGradients;
        public override Tensor Bias => _bias;
        public override Tensor BiasGradients => _biasGradients;

        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (DenseLayer)b;
            var equals = true;
            equals &= Utils.Equals(_units, other._units, id + ":_units", ref errors);
            equals &= Utils.Equals(_lambdaL2Regularization, other._lambdaL2Regularization, epsilon, id, ref errors);
            equals &= _optimizer.Equals(other._optimizer, epsilon, id + ":Optimizer", ref errors);
            return equals;
        }

        #region serialization
        public override string Serialize()
        {
            return RootSerializer() // 'RootSerializer()' will also serialize layer trainable params
                .Add(nameof(_units), _units)
                .Add(nameof(_lambdaL2Regularization), _lambdaL2Regularization)
                .Add(_optimizer.Serialize())
                .ToString();
        }
        public DenseLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _units = (int)serialized[nameof(_units)];
            _lambdaL2Regularization = (double)serialized[nameof(_lambdaL2Regularization)];

            //trainable params
            var useBias = serialized.ContainsKey(nameof(_bias));
            _weights = (Tensor)serialized[nameof(_weights)];
            _bias = useBias ? (Tensor)serialized[nameof(_bias)]  : null;

            _optimizer = Optimizer.ValueOf(network.Config, serialized);
        }
        #endregion

        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            x.AssertIsNotDisposed();
            _weights.AssertIsNotDisposed();
            y.AssertIsNotDisposed();
            //We compute y = x*Weights+B
            y.Dot(x, _weights);
            if (UseBias)
            {
                _bias.BroadcastAddVectorToOutput(y);
            }
        }

        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);
            int batchSize = dy.Shape[0];

            //we allocate '_weightGradients' tensor
            GetNotInitializedFloatTensor(ref _weightGradients, _weights.Shape, nameof(_weightGradients));

            //we compute dW
            var x = allX[0];
            var multiplier = 1f / batchSize;
            if (Network.Config.TensorFlowCompatibilityMode)
            {
                multiplier = 1f; //used only for tests and parallel run
            }
            _weightGradients.Dot(x, true, dy, false, multiplier, 0);

            //L2 regularization on dW
            if (UseL2Regularization)
            {
                var alpha = 2 * batchSize * (float)_lambdaL2Regularization;
                _weightGradients.Update_Adding_Alpha_X(alpha, _weights);
            }

            if (UseBias)
            {
                //we allocate '_biasGradients' tensor
                GetNotInitializedFloatTensor(ref _biasGradients, _bias.Shape, nameof(_biasGradients));
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
        public override void UpdateWeights(int batchSize, double learningRate)
        {
            if (!Trainable)
            {
                return;
            }
            _optimizer.UpdateWeights(learningRate, batchSize, _weights, _weightGradients, _bias, _biasGradients);

            //no more need of '_weightGradients' and '_biasGradients' : we can free them
            FreeMemory(ref _weightGradients);
            FreeMemory(ref _biasGradients);
        }
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
        public override int[] OutputShape(int batchSize)
        {
            return new[] { batchSize, _units };
        }
        public override int DisableBias()
        {
            int nbDisabledWeights = (_bias?.Count ?? 0);
            var bias = _bias;
            FreeMemory(ref bias);
            _bias = null;
            return nbDisabledWeights;
        }

        public int CategoryCount => _units;

        public override string ToString()
        {
            var result = LayerName+": "+ShapeChangeDescription();
            if (UseL2Regularization)
            {
                result += " with L2Regularization[lambdaValue=" + _lambdaL2Regularization + "]";
            }
            result += " " + _weights+ " " + _bias + " (" +MemoryDescription()+")";
            return result;
        }
        public override void Dispose()
        {
            base.Dispose();
            _optimizer?.Dispose();
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

        protected override List<Tensor> TrainableTensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { _weights, _bias};
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        protected override List<Tensor> NonTrainableTensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor>();
                if (_optimizer != null)
                {
                    result.AddRange(_optimizer.EmbeddedTensors);
                }
                result.RemoveAll(t=> t == null);
                return result;
            }
        }
    }
}
