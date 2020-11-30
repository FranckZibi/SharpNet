using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;

// ReSharper disable InconsistentNaming

namespace SharpNet.Layers
{
    /// <summary>
    /// input x shape  :     (batchSize, timeSteps, _inputSize )
    /// output y shape  :    (batchSize, _hiddenSize )
    /// </summary>
    public sealed class SimpleRnnLayerCPU : Layer
    {
        #region Fields
        private readonly int _inputSize;                 // = Features
        private readonly int _hiddenSize;                // = Units
        private readonly bool _returnSequences;

        // ReSharper disable once CollectionNeverUpdated.Local
        /// <summary>
        /// vector of length 'timeSteps_x'
        /// a_t[t] = hidden state at time step 't'          (batchSize, _hiddenSize)
        ///        = tanh( x_t[t]*Weights_ax + a_t[t-1]*Weights_aa)
        ///x_t[t]:                                          (batchSize, _inputSize)
        /// </summary>
        private readonly List<Tensor> a_t = new List<Tensor>();
        #region trainable parameters
        /// <summary>
        /// Weight matrix multiplying the input
        /// Shape:      (_inputSize, _hiddenSize) 
        /// </summary>
        public Tensor Weights_ax;
        /// <summary>
        /// Weight matrix multiplying the hidden state
        /// Shape:      (_hiddenSize, _hiddenSize) 
        /// </summary>
        public Tensor Weights_aa;
        /// <summary>
        /// Bias
        /// shape:      (1, _hiddenSize) 
        /// </summary>
        private Tensor _bias;
        #endregion
        #region gradients
        /// <summary>
        /// same shape as 'Weights_ax'
        /// </summary>
        private Tensor _weights_ax_gradient;
        /// <summary>
        /// same shape as 'Weights_aa'
        /// </summary>
        private Tensor _weights_aa_gradient;
        /// <summary>
        /// same shape as 'Bias'
        /// Can be null if bias has been disabled
        /// </summary>
        [CanBeNull] private Tensor _biasGradients;
        #endregion

        private Tensor x_at_t_buffer;                   //(_timeSteps_x, _inputSize)
        private Tensor a_buffer1;                       //(batchSize, _hiddenSize)          
        private Tensor a_buffer2;                       //(batchSize, _hiddenSize)
        private Tensor a_init;                          //(batchSize, _hiddenSize)
        /// <summary>
        /// Adam or SGD optimizer or Vanilla SGD for Weights_ax matrix and Bias_a
        /// </summary>
        private readonly Optimizer _optimizer_AX_Bias;
        /// <summary>
        /// Adam or SGD optimizer or Vanilla SGD for Weights_aa matrix
        /// </summary>
        private readonly Optimizer _optimizer_AA;
        #endregion

        #region constructor
        public SimpleRnnLayerCPU(int hiddenSize, bool returnSequences, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            _inputSize = PrevLayer.OutputShape(1)[2];
            _hiddenSize = hiddenSize;
            _returnSequences = returnSequences;
            Trainable = trainable;


            //trainable params
            Weights_ax = GetFloatTensor(new[] { _inputSize, _hiddenSize });
            Weights_aa = GetFloatTensor(new[] { _hiddenSize, _hiddenSize });
            _bias = GetFloatTensor(new[] { 1, _hiddenSize });

            //gradients
            _weights_ax_gradient = GetFloatTensor(Weights_ax.Shape);
            _weights_aa_gradient = GetFloatTensor(Weights_aa.Shape);
            _biasGradients = GetFloatTensor(_bias.Shape);

            _optimizer_AX_Bias = GetOptimizer(Weights_ax.Shape, _bias?.Shape);
            _optimizer_AA = GetOptimizer(Weights_aa.Shape, null);
            ResetParameters(false);
        }
        #endregion

        #region forward and backward propagation

        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            if (_returnSequences)
            {
                throw new NotImplementedException("returnSequences is not supported");
            }
            var batchSize = x.Shape[0];                 //x.Shape[0] : batch size : number of sentences 
            int timeSteps = x.Shape[1];                 //x.Shape[1] : number of words in each sentence
            Debug.Assert(x.Shape[2] == _inputSize);      //x.Shape[2] : number of distinct words (_inputSize)
            var aShape = new[] { batchSize, _hiddenSize };
            var xShape = new[] { batchSize, _inputSize };
            GetFloatTensor(ref x_at_t_buffer, xShape);
            GetFloatTensor(ref a_buffer1, aShape);
            GetFloatTensor(ref a_buffer2, aShape);
            GetFloatTensor(ref a_init, aShape);
            a_init.ZeroMemory();

            for (int t = 0; t < timeSteps; ++t)
            {
                if (a_t.Count <= t)
                {
                    a_t.Add(GetFloatTensor(aShape));
                }
                var tmp = a_t[t];
                GetFloatTensor(ref tmp, aShape);

                x.From_NCH_to_NH(x_at_t_buffer, t);

                //a_t[t] : tanh( x_t[t]*Weights_ax + a_t[t-1]*Weights_aa + Bias_a )
                a_buffer1.Dot(x_at_t_buffer, Weights_ax);
                var a_prev = (t == 0) ? a_init : a_t[t - 1];
                a_buffer2.Dot(a_prev, Weights_aa);
                a_buffer1.AddTensor(1, a_buffer2, 1);
                _bias.BroadcastAddVectorToOutput(a_buffer1);
                a_buffer1.ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_TANH, null, a_t[t]);
            }

            a_t.Last().CopyTo(y);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];                // x shape  :     (batchSize, timeSteps, _inputSize)
            var batchSize = x.Shape[0];
            var timeSteps = x.Shape[1];
            Debug.Assert(_inputSize == x.Shape[2]);
            var aShape = dy.Shape;          // a shape  :     (batchSize, _hiddenSize )
            Debug.Assert(a_t.Count == timeSteps);

            _biasGradients?.ZeroMemory();
            _weights_aa_gradient.ZeroMemory();
            _weights_ax_gradient.ZeroMemory();

            GetFloatTensor(ref a_buffer1, aShape);
            GetFloatTensor(ref a_buffer2, aShape);
            var da_t = GetFloatTensor(new []{1, _hiddenSize});
            var _1_minus_aSquare = GetFloatTensor(da_t.Shape);
            var _1_vector = GetFloatTensor(da_t.Shape);
            _1_vector.SetValue(1f);
            var tmpBuffer = GetFloatTensor(da_t.Shape);
            var tmpWeight_aa = GetFloatTensor(_weights_aa_gradient.Shape);

            for (int batchId = 0; batchId < batchSize; ++batchId)
            {
                var x_batchId = x.ElementSlice(batchId);
                x_batchId.Reshape(x.Shape.Skip(1).ToArray());
                for (int t = timeSteps - 1; t >= 0; --t)
                {
                    var at = a_t[t].ElementSlice(batchId);
                    var xt = x_batchId.ElementSlice(t);

                    //we compute _1_minus_aSquare = 1-a[t]^2
                    at.CopyTo(_1_minus_aSquare);
                    _1_minus_aSquare.Update_Multiply_By_x(at);
                    _1_minus_aSquare.AddTensor(1f, _1_vector, -1f);

                    //We compute da[t]
                    if (t == timeSteps - 1)
                    {
                        //da[t] = dy*(1-a[t]^2)
                        _1_minus_aSquare.CopyTo(da_t);
                        da_t.Update_Multiply_By_x(dy.ElementSlice(batchId));
                    }
                    else
                    {
                        //da[t] = (Weights_aa dot da[t+1]) * (1-a[t]^2)
                        da_t.CopyTo(tmpBuffer); //da_t+1
                        da_t.Dot(Weights_aa, false, tmpBuffer, true, 1f, 0f);
                        da_t.Update_Multiply_By_x(_1_minus_aSquare);
                    }

                    //we update the Bias Gradient
                    // _biasGradients += da[t]
                    _biasGradients?.Update_Adding_Alpha_X(1, da_t);

                    //we update _weights_ax_gradient
                    //_weights_ax_gradient += da[t] * x[t]
                    da_t.CopyTo(tmpBuffer);
                    tmpBuffer.Update_Multiply_By_x(xt);
                    _weights_ax_gradient.Update_Adding_Alpha_X(1, tmpBuffer);

                    if (t >= 1)
                    {
                        //we update _weights_aa_gradient
                        var a_t_1 = a_t[t-1].ElementSlice(batchId);
                        tmpWeight_aa.Dot(a_t_1, true, da_t, false, 1f, 0f);
                        //_weights_aa_gradient += a[t-1].T * da[t].T
                        _weights_aa_gradient.Update_Adding_Alpha_X(1, tmpWeight_aa);
                    }
                }
            }

            FreeFloatTensor(ref da_t);
            FreeFloatTensor(ref _1_minus_aSquare);
            FreeFloatTensor(ref tmpBuffer);
            FreeFloatTensor(ref tmpWeight_aa);
            FreeFloatTensor(ref _1_vector);
        }
        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            var prevLayerNX = PrevLayer.n_x;
            Weights_ax.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            Weights_aa.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            _bias?.ZeroMemory();
        }
        #endregion


        #region parameters and gradients
        public override Tensor Weights => throw new Exception("should never be called");
        public override Tensor WeightGradients => Weights_ax;
        public override Tensor Bias => _bias;
        public override Tensor BiasGradients => _biasGradients;
        protected override Optimizer Optimizer => throw new Exception("should never be called");

        public override List<Tuple<Tensor, string>> Parameters
        {
            get
            {
                var result = new List<Tuple<Tensor, string>>
                 {
                     Tuple.Create(Weights_ax, "Weights_ax"),
                     Tuple.Create(Weights_aa, "Weights_aa"),
                     Tuple.Create(_bias, "Bias_a"),
                 };
                result.RemoveAll(t => t.Item1 == null);
                return result;
            }
        }
        public override List<Tensor> ParameterGradients
        {
            get
            {
                var result = new List<Tensor>
                {
                    _weights_ax_gradient,
                    _weights_aa_gradient,
                    _biasGradients,
                };
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        #region Multi GPU Support
        public override void ReplaceParameters(List<Tensor> newParameters)
        {
            FreeFloatTensor(ref Weights_ax);
            Weights_ax = newParameters[0];
            FreeFloatTensor(ref Weights_aa);
            Weights_aa = newParameters[1];
            if (_bias != null)
            {
                Debug.Assert(newParameters.Count == 3);
                FreeFloatTensor(ref _bias);
                _bias = newParameters[2];
            }
            else
            {
                Debug.Assert(newParameters.Count == 2);
            }
        }
        public override void ReplaceGradients(List<Tensor> newGradients)
        {
            FreeFloatTensor(ref _weights_ax_gradient);
            _weights_ax_gradient = newGradients[0];
            FreeFloatTensor(ref _weights_aa_gradient);
            _weights_aa_gradient = newGradients[1];
            if (_biasGradients != null)
            {
                Debug.Assert(newGradients.Count == 3);
                FreeFloatTensor(ref _biasGradients);
                _biasGradients = newGradients[2];
            }
            else
            {
                Debug.Assert(newGradients.Count == 2);
            }
        }
        #endregion

        public override void UpdateWeights(int batchSize, double learningRate)
        {
            Debug.Assert(Network.IsMaster);
            Debug.Assert(Weights_ax.SameShape(_weights_ax_gradient));
            Debug.Assert(Weights_aa.SameShape(_weights_aa_gradient));
            Debug.Assert(_bias == null || _bias.SameShape(_biasGradients));
            if (Trainable)
            {
                _optimizer_AX_Bias.UpdateWeights(learningRate, batchSize, Weights_ax, _weights_ax_gradient, _bias, _biasGradients);
                _optimizer_AA.UpdateWeights(learningRate, batchSize, Weights_aa, _weights_aa_gradient, null, null);
            }
        }
        public override int DisableBias()
        {
            int nbDisabledWeights = (_bias?.Count ?? 0);
            FreeFloatTensor(ref _bias);
            FreeFloatTensor(ref _biasGradients);
            return nbDisabledWeights;
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_hiddenSize), _hiddenSize)
                .Add(nameof(_returnSequences), _returnSequences)
                .ToString();
        }
        public static SimpleRnnLayerCPU Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new SimpleRnnLayerCPU((int)serialized[nameof(_hiddenSize)],
                (bool)serialized[nameof(_returnSequences)],
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            if (_returnSequences)
            {
                int timeSteps = PrevLayer.OutputShape(1)[1];
                return new [] {batchSize, timeSteps, _hiddenSize};
            }
            //only the last output
            return new[] { batchSize, _hiddenSize };
        }
    }
}
