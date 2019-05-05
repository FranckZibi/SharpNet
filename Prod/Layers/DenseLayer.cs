using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Optimizers;

namespace SharpNet
{
    public sealed class DenseLayer : Layer
    {
        #region Private fields
        public Tensor Weights { get; }                      // (prevLayer.n_x, n_x)
        public Tensor WeightGradients { get; }              // same as 'Weights'
        public Tensor Bias { get; private set; }            // (1, n_x) 
        public Tensor BiasGradients { get; private set; }   // same as 'Bias'
        public override Tensor y { get; protected set; }    // (batchSize, n_x)
        public override Tensor dy { get; protected set; }   // same as 'y'
        private readonly int _n_x;
        private readonly double _lambdaL2Regularization;              //regularization hyperparameter. 0 if no L2 regularization
        private bool UseBias => Bias!=null;
        private readonly Optimizer _optimizer;              //Adam or SGD optimizer or Vanilla SGF
        #endregion

        private bool UseL2Regularization => _lambdaL2Regularization > 0.0;

        public DenseLayer(int n_x, double lambdaL2Regularization, Network network) : base(network)
        {
            _n_x = n_x;
            _lambdaL2Regularization = lambdaL2Regularization;
            Weights = Network.NewNotInitializedTensor(new[] { PrevLayer.n_x, _n_x }, Weights, nameof(Weights));
            WeightGradients = Network.NewNotInitializedTensor(Weights.Shape, WeightGradients, nameof(WeightGradients));
            Bias = Network.NewNotInitializedTensor(new[] {1,  _n_x }, Bias, nameof(Bias));
            BiasGradients = Network.NewNotInitializedTensor(Bias.Shape, BiasGradients, nameof(BiasGradients));
            _optimizer = Network.GetOptimizer(Weights.Shape, Bias.Shape);
            ResetWeights(false);
            Debug.Assert(WeightGradients.SameShape(Weights));
            Debug.Assert(Bias.SameShape(BiasGradients));
        }
        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (DenseLayer)b;
            var equals = true;
            equals &= Utils.Equals(_n_x, other._n_x, id + ":LayerIndex", ref errors);
            equals &= Utils.Equals(_lambdaL2Regularization, other._lambdaL2Regularization, epsilon, id, ref errors);
            equals &= _optimizer.Equals(other._optimizer, epsilon, id + ":Optimizer", ref errors);
            return equals;
        }
        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_n_x), _n_x)
                .Add(nameof(_lambdaL2Regularization), _lambdaL2Regularization)
                .Add(Weights).Add(WeightGradients)
                .Add(Bias).Add(BiasGradients)
                .Add(_optimizer.Serialize())
                .ToString();
        }
        public DenseLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _n_x = (int)serialized[nameof(_n_x)];
            _lambdaL2Regularization = (double)serialized[nameof(_lambdaL2Regularization)];
            Weights = (Tensor)serialized[nameof(Weights)];
            WeightGradients = (Tensor)serialized[nameof(WeightGradients)];
            _optimizer = Optimizer.ValueOf(network.Config, serialized);
            Bias = serialized.TryGet<Tensor>(nameof(Bias));
            BiasGradients = serialized.TryGet<Tensor>(nameof(BiasGradients));
        }
        #endregion
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            var x = PrevLayer.y;
            //We compute y = x*Weights+B
            y.Dot(x, Weights);
            if (UseBias)
            {
                Bias.BroadcastAddVectorToOutput(y);
            }
        }

        public override void BackwardPropagation()
        {
            //At this stage, we already know dy
            //we want to compute dx (= prevLayer.dy)  by backward propagation
            int batchSize = y.Shape[0];
            Debug.Assert(y.SameShape(dy));

            //we update dy if necessary (shortcut connection to a futur layer)
            Update_dy_With_GradientFromShortcutIdentityConnection(); 

            //we compute dW
            var x = PrevLayer.y;
            var multiplier = 1.0 / batchSize;
            if (Network.Config.ForceTensorflowCompatibilityMode)
            {
                multiplier = 1.0; //used only for tests and parallel run
            }
            WeightGradients.Dot(x, true, dy, false, multiplier, 0.0);

            //L2 regularization on dW
            if (UseL2Regularization)
            {
                var alpha = 2 * batchSize * _lambdaL2Regularization;
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
            var dx = PrevLayer.dy;
            dx.Dot(dy, false, Weights, true, 1.0, 0.0);
        }
        public override void UpdateWeights(double learningRate)
        {
            var batchSize = y.Shape[0];
            _optimizer.UpdateWeights(learningRate, batchSize, Weights, WeightGradients, Bias, BiasGradients);
        }
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            Weights.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, Math.Sqrt(2.0 / PrevLayer.n_x) /*stdDev*/);
            WeightGradients.ZeroMemory();
            Bias?.ZeroMemory();
            BiasGradients?.ZeroMemory();
            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }
        public override int[] OutputShape(int batchSize)
        {
            return new[] { batchSize, _n_x };
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
        public override string ToString()
        {
            string layerName = SummaryName();
            string result = layerName+": "+ShapeChangeDescription();
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
        public override int TotalParams => (Weights?.Count??0) + (Bias?.Count??0);

        public override List<Tensor> TensorsIndependantOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { Weights, WeightGradients, Bias, BiasGradients };
                if (_optimizer != null)
                {
                    result.AddRange(_optimizer.EmbeddedTensors);
                }
                result.RemoveAll(x => x == null);
                return result;
            }
        }
    }
}
