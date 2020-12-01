using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public sealed class SimpleRnnLayerGPU : RecurrentLayer
    {
        #region constructor
        public SimpleRnnLayerGPU(int units, bool returnSequences, bool trainable, Network network, string layerName) : 
            base(units, cudnnRNNMode_t.CUDNN_RNN_TANH, cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS, returnSequences, trainable, network, layerName)
        {
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(HiddenSize), HiddenSize)
                .Add(nameof(_returnSequences), _returnSequences)
                .ToString();
        }
        public static SimpleRnnLayerGPU Deserialize(IDictionary<string, object> serialized, Network network)
        {
            Debug.Assert(network.UseGPU);
            return new SimpleRnnLayerGPU((int)serialized[nameof(HiddenSize)],
                (bool)serialized[nameof(_returnSequences)],
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        private static RNNDescriptor BuildRNNDescriptor(int units, Layer prevLayer)
        {
            int inputSize = prevLayer.OutputShape(1)[2]; // InputSize = Features
            uint auxFlags = 0;
            auxFlags |= CudnnWrapper.CUDNN_RNN_PADDED_IO_ENABLED;

            return new RNNDescriptor(
                cudnnRNNAlgo_t.CUDNN_RNN_ALGO_STANDARD,
                cudnnRNNMode_t.CUDNN_RNN_TANH,
                cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS,
                cudnnDirectionMode_t.CUDNN_UNIDIRECTIONAL,
                cudnnRNNInputMode_t.CUDNN_LINEAR_INPUT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnMathType_t.CUDNN_DEFAULT_MATH,
                inputSize,  /* = features */
                units,      /* hiddenSize */
                units,      /* projSize*/
                1,          /* numLayers */
                0.0,
                auxFlags);
        }
    }
}
