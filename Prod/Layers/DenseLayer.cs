using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Optimizers;

namespace SharpNet
{
    public class DenseLayer : Layer
    {
        #region Private fields
        public Tensor Weights { get; }                      // (prevLayer.n_x, n_x)
        public Tensor WeightGradients { get; }              // same as 'Weights'
        public Tensor Bias { get; }                         // (1, n_x)
        public Tensor BiasGradients { get; }                  // same as '_bias'
        public override Tensor y { get; protected set; }    // (batchSize, n_x)
        public override Tensor dy { get; protected set; }   // same as 'y'
        private readonly int _n_x;
        private readonly double _lambdaL2Regularization;              //regularization hyperparameter. 0 if no L2 regularization
        private readonly Optimizer _optimizer;              //Adam or SGD optimizer or Vanilla SGF
        #endregion

        private bool UseL2Regularization => _lambdaL2Regularization > 0.0;

        public DenseLayer(int n_x, double lambdaL2Regularization, Network network) : base(network)
        {
            _n_x = n_x;
            _lambdaL2Regularization = lambdaL2Regularization;
            Weights = Network.RandomMatrixNormalDistribution(new[] { PrevLayer.n_x, _n_x }, 0.0 /* mean */, Math.Sqrt(2.0 / PrevLayer.n_x) /*stdDev*/, nameof(Weights));
            WeightGradients = Network.NewTensor(Weights.Shape, nameof(WeightGradients));
            Bias = Network.NewTensor(new[] {1,  _n_x }, nameof(Bias));
            BiasGradients = Network.NewTensor(Bias.Shape, nameof(BiasGradients));
            _optimizer = Network.GetOptimizer(Weights.Shape, Bias.Shape);
        }
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_n_x), _n_x)
                .Add(nameof(_lambdaL2Regularization), _lambdaL2Regularization)
                .Add(Weights).Add(WeightGradients).Add(Bias).Add(BiasGradients)
                .Add(_optimizer?.Serialize())
                .ToString();
        }
        public static DenseLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new DenseLayer(serialized, network);
        }
        private DenseLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _n_x = (int)serialized[nameof(_n_x)];
            _lambdaL2Regularization = (double)serialized[nameof(_lambdaL2Regularization)];
            Weights = (Tensor)serialized[nameof(Weights)];
            WeightGradients = (Tensor)serialized[nameof(WeightGradients)];
            Bias = (Tensor)serialized[nameof(Bias)];
            BiasGradients = (Tensor)serialized[nameof(BiasGradients)];
            _optimizer = Optimizer.ValueOf(network.Config, serialized);
        }

        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            var x = PrevLayer.y;
            //We compute y = x*Weights+B
            y.Dot(x, Weights);
            Bias.BroadcastAddVectorToOutput(y);
        }

        public override void BackwardPropagation()
        {
            //At this stage, we already know dy
            //we want to compute dx (= prevLayer.dy)  by backward propagation
            int batchSize = y.Shape[0];
            Debug.Assert(y.SameShape(dy));

            //we update dy if necessary (shortcut connection to a futur layer)
            Update_dy_With_GradientFromShortcutIdentityConnection(); 

            //we compute dW:  prevLayer.n_x * n_x
            var multiplier = 1.0; //!D var multiplier = 1.0 / batchSize;
            var x = PrevLayer.y;
            WeightGradients.Dot(x, true, dy, false, multiplier, 0.0);
            //L2 regularization on dW
            if (UseL2Regularization)
            {
                var alpha = _lambdaL2Regularization / batchSize;
                WeightGradients.Update_Adding_Alpha_X(alpha, Weights);
            }
            Debug.Assert(WeightGradients.SameShape(Weights));

            dy.Compute_BiasGradient_from_dy(BiasGradients);

            Debug.Assert(BiasGradients.SameShape(Bias));

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
            Debug.Assert(Weights.SameShape(WeightGradients));
            Debug.Assert(Bias.SameShape(BiasGradients));
            var batchSize = y.Shape[0];
            _optimizer.UpdateWeights(learningRate, batchSize, Weights, WeightGradients, Bias, BiasGradients);
        }
        public override int[] OutputShape(int batchSize)
        {
            return new[] { batchSize, _n_x };
        }

        public override string SummaryName()
        {
            return (NextLayer!=null&&NextLayer.IsOutputLayer) ? "OutputLayer" : "Dense";
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
        public override int TotalParams
        {
            get
            {
                if (Weights == null)
                {
                    return 0;
                }
                return Weights.Count + Bias.Count;
            }
        }
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
