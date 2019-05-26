using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

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
        private readonly Tensor _adam_VW;                      // same as 'Weights'
        private readonly Tensor _adam_SW;                      // same as 'Weights'
        private readonly Tensor _adam_VB;                      // same as 'Weights Bias'
        private readonly Tensor _adam_SB;                      // same as 'Weight Bias'
        #endregion

        public Adam(Network network, int[] weightShape, int[] biasShapeIfAny) : base(network.Config)
        {
            _adam_VW = network.NewNotInitializedTensor(weightShape, _adam_VW, nameof(_adam_VW));
            _adam_SW = network.NewNotInitializedTensor(weightShape, _adam_SW, nameof(_adam_SW));
            _adam_VB = (biasShapeIfAny == null) ? null : network.NewNotInitializedTensor(biasShapeIfAny, _adam_VB, nameof(_adam_VB));
            _adam_SB = (biasShapeIfAny == null) ? null : network.NewNotInitializedTensor(biasShapeIfAny, _adam_SB, nameof(_adam_SB));
            ZeroMemory();
        }

        public override Optimizer Clone(Network newNetwork) { return new Adam(this, newNetwork); }
        private Adam(Adam toClone, Network newNetwork) : base(newNetwork.Config)
        {
            _timestep = toClone._timestep;
            _adam_VW = toClone._adam_VW?.Clone(newNetwork.GpuWrapper);
            _adam_SW = toClone._adam_SW?.Clone(newNetwork.GpuWrapper);
            _adam_VB = toClone._adam_VB?.Clone(newNetwork.GpuWrapper);
            _adam_SB = toClone._adam_SB?.Clone(newNetwork.GpuWrapper);
        }

        public override bool Equals(Optimizer other, double epsilon, string id, ref string errors)
        {
            if (!Utils.Equals(GetType(), other.GetType(), id + ":GetType", ref errors))
            {
                return false;
            }
            var b = (Adam)other;
            return  Utils.Equals(_timestep, b._timestep, id+ "_timestep", ref errors)
                && _adam_VW.Equals(b._adam_VW, epsilon, id + "_adam_VW", ref errors)
                && _adam_SW.Equals(b._adam_SW, epsilon, id + "_adam_SW", ref errors)
                && _adam_VB.Equals(b._adam_VB, epsilon, id + "_adam_VB", ref errors)
                && _adam_SB.Equals(b._adam_SB, epsilon, id + "_adam_SB", ref errors);
        }

        public override List<Tensor> EmbeddedTensors => new List<Tensor> { _adam_VW, _adam_SW, _adam_SB, _adam_VB };
        public override void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradient)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradient));
            ++_timestep;
            var beta1 = _networkConfig.Adam_beta1;
            var beta2 = _networkConfig.Adam_beta2;
            var epsilon = _networkConfig.Adam_epsilon;
            var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
            weights.UpdateAdamOptimizer(ponderedLearningRate, beta1, beta2, epsilon, weightGradients, _adam_VW, _adam_SW, _timestep);
            bias?.UpdateAdamOptimizer(ponderedLearningRate, beta1, beta2, epsilon, biasGradient, _adam_VB, _adam_SB, _timestep);
        }

        #region serialization
        public override string Serialize()
        {
            return new Serializer()
                .Add(nameof(_timestep), _timestep)
                .Add(_adam_VW).Add(_adam_SW)
                .Add(_adam_VB).Add(_adam_SB)
                .ToString();
        }
        public static Optimizer DeserializeAdam(NetworkConfig networkConfig, IDictionary<string, object> serialized)
        {
            return serialized.ContainsKey(nameof(_adam_VW)) ? new Adam(networkConfig, serialized) : null;
        }
        private Adam(NetworkConfig networkConfig, IDictionary<string, object> serialized) : base(networkConfig)
        {
            _timestep = (int)serialized[nameof(_timestep)];
            _adam_VW = (Tensor)serialized[nameof(_adam_VW)];
            _adam_SW = (Tensor)serialized[nameof(_adam_SW)];
            serialized.TryGet(nameof(_adam_VB), out _adam_VB);
            serialized.TryGet(nameof(_adam_SB), out _adam_SB);
        }



        #endregion

    }
}