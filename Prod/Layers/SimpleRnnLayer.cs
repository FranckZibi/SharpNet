using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
// ReSharper disable InconsistentNaming

namespace SharpNet.Layers
{
    // 'batchSize'   : number of sentences to process in current batch
    // Foreach each timeStep in (1, timeSteps_x)
    //      We'll process all words at position 'timeStep' in all sentences & update the network
    //              input (for each sentence): the hidden state of previous time step + the word at time step 't'
    //              output (for each sentence): the hidden state of time step 't' + the output 'y' at time step 't'
    public sealed class SimpleRnnLayer : Layer
    {
        #region Fields
        private readonly int _timeSteps_x;  //number of words in each sentence
        private readonly int _xLength;      //number of distinct words in the dictionary 
        private readonly int _aLength;      //length of a hidden state for a specific time step / specific sentence
        private readonly int _yLength;
        private readonly bool _returnSequences;
        private Tensor a_init;         //(batchSize, aLength)
        private readonly List<Tensor> a_t = new List<Tensor>();   //vector of length 'timeSteps_x' / each element: (batchSize, aLength)
        //a_t(t) hidden state at time step 't' (batchSize, aLength)
        //a_t(t): tanh( x_t(t)*Weights_ax + a_t(t-1)*Weights_aa + Bias_a)
        //x_t(t): (batchSize, _xLength)
        //Weight matrix multiplying the input
        public readonly Tensor Weights_ax;              // (xLength, aLength) 
        //Weight matrix multiplying the hidden state
        public readonly Tensor Weights_aa;              // (aLength, aLength) 
        //Bias
        public readonly Tensor Bias_a;                  // (1, aLength) 
        //vector of length 'timeSteps_y' / each element: (batchSize, yLength)
        private readonly List<Tensor> y_t = new List<Tensor>();       //y(t) = softmax( a(t)*Weights_ay + Bias_y)
        public readonly Tensor Weights_ay;              // (aLength, yLength) 
        //Bias relating the hidden-state to the output
        public readonly Tensor Bias_y;                  // (1, yLength) 
        private Tensor x_at_t_buffer;
        private Tensor a_buffer1;
        private Tensor a_buffer2 ;
        private Tensor y_buffer1;
        //public override Tensor y
        //{
        //    get
        //    {
        //        if (_returnSequences)
        //        {
        //            throw new NotImplementedException(); //TODO
        //        }
        //        return y_t.Last();
        //    }
        //    protected set => throw new NotImplementedException();
        //}
        #endregion

        //No need to configure the number of x time step : it is always the same as in previous layer number of channels
        public SimpleRnnLayer(int xLength, int aLength, int yLength, bool returnSequences, Network network, string layerName) : base(network, layerName)
        {
            _timeSteps_x = PrevLayer.OutputShape(1)[1];
            _xLength = xLength;
            _aLength = aLength;
            _yLength = yLength;
            _returnSequences = returnSequences;

            Weights_aa = GetFloatTensor(new[] { aLength, aLength });
            Weights_ax = GetFloatTensor(new[] { xLength, aLength });
            Weights_ay = GetFloatTensor(new[] { aLength, yLength });
            Bias_a = GetFloatTensor(new[] { 1, aLength });
            Bias_y = GetFloatTensor(new[] { 1, yLength });
            //a_init = Network.GetFloatTensor(new[] { batchSize, aLength }, a_init, nameof(a_init));
            ResetWeights(false);
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y1, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x_NCH = allX[0];
            var batchSize = x_NCH.Shape[0];                 //N : batch size : number of sentences : _timeSteps_x
            Debug.Assert(x_NCH.Shape[1] == _timeSteps_x);   //C : number of time steps = number of words per sentence
            Debug.Assert(x_NCH.Shape[2] == _xLength);       //H : number of distinct words : _xLength

            var aShape = new[] { batchSize, _aLength };
            var yShape = new[] { batchSize, _yLength };

            var shape_NCH = x_NCH.Shape;
            var shape_NH = new[] { shape_NCH[0], shape_NCH[2] };
            GetFloatTensor(ref x_at_t_buffer, shape_NH);
            GetFloatTensor(ref a_buffer1, aShape);
            GetFloatTensor(ref a_buffer2, aShape);
            GetFloatTensor(ref y_buffer1, yShape);


            for (int t = 0; t < _timeSteps_x; ++t)
            {
                x_NCH.From_NCH_to_NH(x_at_t_buffer, t);
                //a(t): tanh( x(t)*Weights_ax + a(t-1)*Weights_aa + Bias_a)

                a_buffer1.Dot(x_at_t_buffer, Weights_ax);
                var a_prev = (t == 0) ? a_init : a_t[t - 1];
                a_buffer2.Dot(a_prev, Weights_aa);
                a_buffer1.AddTensor(1, a_buffer2, 1);
                Bias_a.BroadcastAddVectorToOutput(a_buffer1);
                a_buffer1.ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_TANH, a_t[t]);

                y_buffer1.Dot(a_t[t], Weights_ay);
                Bias_y.BroadcastAddVectorToOutput(y_buffer1);
                y_buffer1.ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, y_t[t]);
            }
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            throw new NotImplementedException();
        }
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            var prevLayerNX = PrevLayer.n_x;
            Weights_aa.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            Weights_ax.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            Weights_ay.RandomMatrixNormalDistribution(Network.Config.Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            Bias_a?.ZeroMemory();
            Bias_y?.ZeroMemory();
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_timeSteps_x), _timeSteps_x).Add(nameof(_xLength), _xLength)
                .Add(nameof(_aLength), _aLength)
                .Add(nameof(_returnSequences), _returnSequences)
                .Add(nameof(_yLength), _yLength)
                .ToString();
        }
        public SimpleRnnLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _timeSteps_x = (int)serialized[nameof(_timeSteps_x)];
            _xLength = (int)serialized[nameof(_xLength)];
            _aLength = (int)serialized[nameof(_aLength)];
            _yLength = (int)serialized[nameof(_yLength)];
            _returnSequences = (bool)serialized[nameof(_returnSequences)];
        }
        #endregion

        public override void AddToOtherNetwork(Network otherNetwork)
        {
            otherNetwork.Layers.Add(new SimpleRnnLayer(_xLength, _aLength, _yLength, _returnSequences, otherNetwork, LayerName));
        }

        public override int[] OutputShape(int batchSize)
        {
            if (_returnSequences)
            {
                return new [] {batchSize, _timeSteps_x, _yLength};
            }
            //only the last output
            return new[] { batchSize, _yLength };
        }
    }
}
