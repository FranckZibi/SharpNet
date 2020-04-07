using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    public sealed class DenseLayer : Layer
    {
        #region Private fields
        public Tensor Weights { get; }                      // (prevLayer.n_x, n_x)
        public Tensor WeightGradients { get; }              // same as 'Weights'

        /// <summary>
        /// (1, n_x)
        /// Can be null if bias has been disabled
        /// </summary>
        public Tensor Bias { get; private set; }
        /// <summary>
        /// same shape as Bias
        /// Can be null if bias has been disabled
        /// </summary>
        public Tensor BiasGradients { get; private set; }
        public override Tensor y { get; protected set; }    // (batchSize, n_x)
        /// <summary>
        /// dimensionality of the output space
        /// </summary>
        private readonly int _units;
        /// <summary>
        /// regularization hyper parameter. 0 if no L2 regularization
        /// </summary>
        public readonly double _lambdaL2Regularization;
        private bool UseBias => Bias!=null;
        private readonly Optimizer _optimizer;              //Adam or SGD optimizer or Vanilla SGF
        #endregion

        private bool UseL2Regularization => _lambdaL2Regularization > 0.0;

        public DenseLayer(int units, double lambdaL2Regularization, Network network, string layerName) : base(network, layerName)
        {
            _units = units;
            _lambdaL2Regularization = lambdaL2Regularization;

            //trainable params
            Weights = Network.NewNotInitializedFloatTensor(new[] { PrevLayer.n_x, _units }, nameof(Weights));
            Bias = Network.NewNotInitializedFloatTensor(new[] {1,  _units }, nameof(Bias));
            _optimizer = Network.GetOptimizer(Weights.Shape, Bias.Shape);
            ResetWeights(false);

            //non trainable params
            WeightGradients = Network.NewNotInitializedFloatTensor(Weights.Shape, nameof(WeightGradients));
            BiasGradients = Network.NewNotInitializedFloatTensor(Bias.Shape, nameof(BiasGradients));

            Debug.Assert(WeightGradients.SameShape(Weights));
            Debug.Assert(Bias.SameShape(BiasGradients));
        }

        public override Layer Clone(Network newNetwork) { return new DenseLayer(this, newNetwork); }
        private DenseLayer(DenseLayer toClone, Network newNetwork) : base(toClone, newNetwork)
        {
            _units = toClone._units;
            _lambdaL2Regularization = toClone._lambdaL2Regularization;
            Weights = toClone.Weights?.Clone(newNetwork.GpuWrapper);
            WeightGradients = toClone.WeightGradients?.Clone(newNetwork.GpuWrapper);
            Bias = toClone.Bias?.Clone(newNetwork.GpuWrapper);
            BiasGradients = toClone.BiasGradients?.Clone(newNetwork.GpuWrapper);
            _optimizer = toClone._optimizer?.Clone(newNetwork);
        }
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
            var useBias = serialized.ContainsKey(nameof(Bias));
            Weights = (Tensor)serialized[nameof(Weights)];
            Bias = useBias ? (Tensor)serialized[nameof(Bias)]  : null;

            _optimizer = Optimizer.ValueOf(network.Config, serialized);

            //non trainable params
            WeightGradients = Network.NewNotInitializedFloatTensor(Weights.Shape, nameof(WeightGradients));
            BiasGradients = useBias ? Network.NewNotInitializedFloatTensor(Bias.Shape, nameof(BiasGradients)) : null;
        }
        #endregion

        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PrevLayer.y;

            x.AssertIsNotDisposed();
            Weights.AssertIsNotDisposed();
            y.AssertIsNotDisposed();

            //We compute y = x*Weights+B
            y.Dot(x, Weights);
            if (UseBias)
            {
                Bias.BroadcastAddVectorToOutput(y);
            }
        }

        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);
            int batchSize = y.Shape[0];
            Debug.Assert(y.SameShape(dy));

            //we compute dW
            var x = PrevLayer.y;
            var multiplier = 1f / batchSize;
            if (Network.Config.TensorFlowCompatibilityMode)
            {
                multiplier = 1f; //used only for tests and parallel run
            }
            WeightGradients.Dot(x, true, dy, false, multiplier, 0);

            //L2 regularization on dW
            if (UseL2Regularization)
            {
                var alpha = 2 * batchSize * (float)_lambdaL2Regularization;
                WeightGradients.Update_Adding_Alpha_X(alpha, Weights);
            }

            if (UseBias)
            {
                dy.Compute_BiasGradient_from_dy(BiasGradients);
            }

            //no need to compute dx (= PrevLayer.dy) if previous Layer it is the input layer
            if (PrevLayer.IsInputLayer)
            {
                return;
            }

            // we compute dx = dy * Weights.T
            dx[0].Dot(dy, false, Weights, true, 1, 0);
        }
        public override void UpdateWeights(double learningRate)
        {
            if (!Trainable)
            {
                return;
            }
            var batchSize = y.Shape[0];
            _optimizer.UpdateWeights(learningRate, batchSize, Weights, WeightGradients, Bias, BiasGradients);
        }
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            //trainable params
            Weights.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, Math.Sqrt(2.0 / PrevLayer.n_x) /*stdDev*/);
            Bias?.ZeroMemory();

            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }

            // non trainable params : no need to reset them
            //WeightGradients.ZeroMemory();
            //BiasGradients?.ZeroMemory();
        }
        public override int[] OutputShape(int batchSize)
        {
            return new[] { batchSize, _units };
        }
        public override int DisableBias()
        {
            int nbDisabledWeights = (Bias?.Count ?? 0) + (BiasGradients?.Count ?? 0);
            Bias?.Dispose();
            Bias = null;
            BiasGradients?.Dispose();
            BiasGradients = null;
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
            result += " " + Weights+ " " + Bias + " (" +MemoryDescription()+")";
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
            h5FileDataset[DatasetNameToDatasetPath("kernel:0")].CopyTo(Weights);

            //var biasCpuTensor = (CpuTensor<float>)h5FileDataset[biasDatasetPath];
            //var reshapedBiasCpuTensor = biasCpuTensor.WithNewShape(new[] {1, biasCpuTensor.Shape[0]});
            h5FileDataset[DatasetNameToDatasetPath("bias:0")].CopyTo(Bias);
        }

        protected override List<Tensor> TrainableTensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { Weights, Bias};
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        protected override List<Tensor> NonTrainableTensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { WeightGradients, BiasGradients };
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
