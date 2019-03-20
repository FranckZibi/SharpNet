using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Optimizers;

namespace SharpNet
{
    public class NetworkConfig
    {
        #region fields
        public LossFunctionEnum LossFunction { get; set;} = LossFunctionEnum.CategoricalCrossentropy;
        public GPUWrapper GpuWrapper { get; }
        public double Adam_beta1 { get; private set; }
        public double Adam_beta2 { get; private set; }
        public double Adam_epsilon { get; private set; }
        public double SGD_momentum { get; private set; }
        public double SGD_decay { get; private set; }
        /// <summary>
        /// minimum value for the learning rate
        /// </summary>
        public double MinimumLearningRate { get; } = 0.5*1e-6;
        public bool SGD_usenesterov { get; private set; }
        public Random Rand { get; }
        public Logger Logger { get; set; } = Logger.ConsoleLogger;
        public bool UseDoublePrecision { get; set; }
        public bool RandomizeOrder { get; set; } = true;
        public bool ProfileApplication { get; } = true;
        public string AutoSavePath { get; set; } = System.IO.Path.GetTempPath();
        public int AutoSaveIntervalInMinuts { get; set; } = 15;

        #endregion

        public NetworkConfig(bool useGPU = true)
        {
            GpuWrapper = useGPU ? GPUWrapper.Default : null;
            Rand = new Random(0);
        }
        public bool UseGPU => GpuWrapper != null;
        public int TypeSize => UseDoublePrecision ? 8 : 4;
        public NetworkConfig WithAdam(double _beta1 = 0.9, double _beta2 = 0.999)
        {
            Debug.Assert(_beta1 >= 0);
            Debug.Assert(_beta1 <= 1.0);
            Debug.Assert(_beta2 >= 0);
            Debug.Assert(_beta2 <= 1.0);
            Adam_beta1 = _beta1;
            Adam_beta2 = _beta2;
            Adam_epsilon = 1e-8;
            OptimizerType = Optimizer.OptimizationEnum.Adam;
            return this;
        }

        public NetworkConfig WithSGD(double momentum= 0.9, double decay = 0.0, bool useNesterov = true)
        {
            Debug.Assert(momentum >= 0);
            Debug.Assert(momentum <= 1.0);
            Debug.Assert(decay >= 0);
            SGD_momentum = momentum;
            SGD_decay = decay;
            SGD_usenesterov = useNesterov;
            OptimizerType = Optimizer.OptimizationEnum.SGD;
            return this;
        }
        public Optimizer.OptimizationEnum OptimizerType { get; private set; } = Optimizer.OptimizationEnum.VanillaSGD;
        //set the path for saving the network (by default the temp folder)
        public void AutoSave(string autoSavePath, int intervalInMinuts = 15)
        {
            AutoSavePath = autoSavePath;
            AutoSaveIntervalInMinuts = intervalInMinuts;
        }

        #region serialization
        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(LossFunction), (int)LossFunction).Add(nameof(OptimizerType), (int)OptimizerType)
                .Add(nameof(Adam_beta1), Adam_beta1).Add(nameof(Adam_beta2), Adam_beta2).Add(nameof(Adam_epsilon), Adam_epsilon)
                .Add(nameof(SGD_momentum), SGD_momentum).Add(nameof(SGD_decay), SGD_decay).Add(nameof(SGD_usenesterov), SGD_usenesterov)
                .Add(nameof(UseDoublePrecision), UseDoublePrecision)
                .Add(nameof(RandomizeOrder), RandomizeOrder)
                .Add(nameof(ProfileApplication), ProfileApplication)
                .Add(nameof(AutoSaveIntervalInMinuts), AutoSaveIntervalInMinuts).Add(nameof(AutoSavePath), AutoSavePath)
                .Add(nameof(UseGPU), UseGPU)
                .Add(nameof(MinimumLearningRate), MinimumLearningRate)
                .Add(Logger.Serialize())
                .ToString();
        }
        public static NetworkConfig ValueOf(IDictionary<string, object> serialized)
        {
            return new NetworkConfig(serialized);
        }
        private NetworkConfig(IDictionary<string, object> serialized)
        {
            LossFunction = (LossFunctionEnum)serialized[nameof(LossFunction)];
            OptimizerType = (Optimizer.OptimizationEnum)serialized[nameof(OptimizerType)];
            Adam_beta1 = (double)serialized[nameof(Adam_beta1)];
            Adam_beta2 = (double)serialized[nameof(Adam_beta2)];
            Adam_epsilon = (double)serialized[nameof(Adam_epsilon)];
            SGD_momentum= (double)serialized[nameof(SGD_momentum)];
            SGD_decay = (double)serialized[nameof(SGD_decay)];
            SGD_usenesterov = (bool)serialized[nameof(SGD_usenesterov)];
            UseDoublePrecision = (bool)serialized[nameof(UseDoublePrecision)];
            RandomizeOrder = (bool)serialized[nameof(RandomizeOrder)];
            ProfileApplication = (bool)serialized[nameof(ProfileApplication)];
            AutoSaveIntervalInMinuts = (int)serialized[nameof(AutoSaveIntervalInMinuts)];
            AutoSavePath = (string)serialized[nameof(AutoSavePath)];
            var useGPU = (bool)serialized[nameof(UseGPU)];
            MinimumLearningRate = (double)serialized[nameof(MinimumLearningRate)];
            GpuWrapper = useGPU ? GPUWrapper.Default : null;
            Logger = Logger.ValueOf(serialized);
            Rand = new Random(0);
        }
        #endregion

        //conventions: 
        //  the output layer has a shape of (N, C) where:
        //      'N' is the number of batches
        //      'C' the number of distinct categories
        public enum LossFunctionEnum
        {
            //To be used with sigmoid activation layer.
            //In a single row, each value will be in [0,1] range
            //Support of multi labels (one element can belong to several categories at the same time)
            BinaryCrossentropy,

            //To be used with softmax activation layer.
            //In a single row, each value will be in [0,1] range, and the sum of all values wil be equal to 1.0 (= 100%)
            //Do not support multi labels (each element can belong to exactly 1 category)
            CategoricalCrossentropy
        }
    }
}
