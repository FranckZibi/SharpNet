using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    //beta1 = 0., beta2 = 0.: "vanilla" SGD
    //beta1 = 0.9, beta2 = 0.: "Classic momentum"
    //beta1 = 0.0, beta2 = 0.2: RMS prop
    //beta1 = 0.999, beta2 = 0.9: Classic Adam
    // Adam with beta1=1 is equivalent to RMSProp with momentum=0. 
    //The argument beta2 of Adam and the argument decay of RMSProp are the same
    public class Adam : Optimizer
    {
        #region private fields
        private int _timestep = 1;
        private readonly Tensor adam_vW;                      // same as 'Weights'
        private readonly Tensor adam_sW;                      // same as 'Weights'
        private readonly Tensor adam_vB;                      // same as 'Weights Bias'
        private readonly Tensor adam_sB;                      // same as 'Weight Bias'
        #endregion

        public Adam(Network network, int[] weightShape, int[] biasShape) : base(network.Config)
        {
            adam_vW = network.NewTensor(weightShape, nameof(adam_vW));
            adam_sW = network.NewTensor(weightShape, nameof(adam_sW));
            adam_vB = network.NewTensor(biasShape, nameof(adam_vB));
            adam_sB = network.NewTensor(biasShape, nameof(adam_sB));
        }
        public override List<Tensor> EmbeddedTensors => new List<Tensor> { adam_vW, adam_sW, adam_sB, adam_vB };
        public override void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradient)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradient));
            ++_timestep;
            var beta1 = _networkConfig.Adam_beta1;
            var beta2 = _networkConfig.Adam_beta2;
            var epsilon = _networkConfig.Adam_epsilon;
            weights.UpdateAdamOptimizer(learningRate, beta1, beta2, epsilon, weightGradients, adam_vW, adam_sW, _timestep);
            bias?.UpdateAdamOptimizer(learningRate, beta1, beta2, epsilon, biasGradient, adam_vB, adam_sB, _timestep);
        }

        #region serialization
        public override string Serialize()
        {
            return new Serializer()
                .Add(nameof(_timestep), _timestep)
                .Add(adam_vW).Add(adam_sW).Add(adam_vB).Add(adam_sB)
                .ToString();
        }
        public static Optimizer DeserializeAdam(NetworkConfig networkConfig, IDictionary<string, object> serialized)
        {
            return serialized.ContainsKey(nameof(adam_vW)) ? new Adam(networkConfig, serialized) : null;
        }
        private Adam(NetworkConfig networkConfig, IDictionary<string, object> serialized) : base(networkConfig)
        {
            _timestep = (int)serialized[nameof(_timestep)];
            adam_vW = (Tensor)serialized[nameof(adam_vW)];
            adam_sW = (Tensor)serialized[nameof(adam_sW)];
            adam_vB = (Tensor)serialized[nameof(adam_vB)];
            adam_sB = (Tensor)serialized[nameof(adam_sB)];
        }
        #endregion

    }
}