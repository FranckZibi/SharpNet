using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.Data;
using SharpNet.Optimizers;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Networks
{
    public class NetworkConfig
    {
        #region fields

        #region learning rate scheduler fields
        public enum LearningRateSchedulerEnum { Cifar10ResNet, Cifar10DenseNet, OneCycle, CyclicCosineAnnealing, Cifar10WideResNet}
        public LearningRateSchedulerEnum LearningRateSchedulerType { get; set; } = LearningRateSchedulerEnum.Cifar10ResNet;
        public int CyclicCosineAnnealing_nbEpochsInFirstRun { get; private set; } = 10;
        public int CyclicCosineAnnealing_nbEpochInNextRunMultiplier { get; private set; } = 2;
        //for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        public int OneCycle_DividerForMinLearningRate { get; set; } = 10;
        public double OneCycle_PercentInAnnealing { get; set; } = 0.2;
        public bool DisableReduceLROnPlateau { get; set; }
        public bool DivideBy10OnPlateau { get; set; } = true; // 'true' : validated on 19-apr-2019: +20 bps
        public bool LinearLearningRate { get; set; }
        #endregion

        public LossFunctionEnum LossFunction { get; set;} = LossFunctionEnum.CategoricalCrossentropy;
        public double Adam_beta1 { get; private set; }
        public double Adam_beta2 { get; private set; }
        public double Adam_epsilon { get; private set; }
        public double SGD_momentum { get; private set; }
        public double lambdaL2Regularization { get; set; }
        /// <summary>
        /// minimum value for the learning rate
        /// </summary>
        public double MinimumLearningRate { get; } = 1e-7;
        public bool SGD_usenesterov { get; private set; }
        public Random Rand { get; }
        public Logger Logger { get; set; } = Logger.ConsoleLogger;
        public bool UseDoublePrecision { get; set; }
        public bool RandomizeOrder { get; set; } = true;
        /// <summary>
        /// true if we should use exactly the same conventions then Tensorflow
        /// used only for // run and testing
        /// </summary>
        public bool ForceTensorflowCompatibilityMode { get; set; }
        /// <summary>
        /// true if we want t display statistics about the weights tensors.
        /// Used only for debugging 
        /// </summary>
        public bool DisplayTensorContentStats{ get; set; }
        public bool ProfileApplication { get; } = true;

        /// <summary>
        /// Interval in minuts for saving the network
        /// If less then 0
        ///     => this option will be disabled
        /// If == 0
        ///     => the network will be saved after each iteration
        /// </summary>
        public int AutoSaveIntervalInMinuts { get; set; } = 90;
        public bool SaveNetworkStatsAfterEachEpoch { get; set; }
        public bool SaveLossAfterEachMiniBatch { get; set; }
        public string LogDirectory { get; } = DefaultLogDirectory;
        public static string DefaultLogDirectory => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpNet");
        #endregion

        public NetworkConfig()
        {
            Rand = new Random(0);
        }


        public bool DisableLogging => ReferenceEquals(Logger, SharpNet.Logger.NullLogger);

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

        public NetworkConfig WithSGD(double momentum = 0.9, bool useNesterov = true)
        {
            Debug.Assert(momentum >= 0);
            Debug.Assert(momentum <= 1.0);
            SGD_momentum = momentum;
            SGD_usenesterov = useNesterov;
            OptimizerType = Optimizer.OptimizationEnum.SGD;
            return this;
        }
        public Optimizer.OptimizationEnum OptimizerType { get; private set; } = Optimizer.OptimizationEnum.VanillaSGD;


        public bool Equals(NetworkConfig other, double epsilon, string id, ref string errors)
        {
            var equals = true;
            equals &= Utils.Equals(LossFunction, other.LossFunction, id + ":LossFunction", ref errors);

            equals &= Utils.Equals(Adam_beta1, other.Adam_beta1, epsilon, id + ":Adam_beta1", ref errors);
            equals &= Utils.Equals(Adam_beta2, other.Adam_beta2, epsilon, id + ":Adam_beta2", ref errors);
            equals &= Utils.Equals(Adam_epsilon, other.Adam_epsilon, epsilon, id + ":Adam_epsilon", ref errors);

            equals &= Utils.Equals(SGD_momentum, other.SGD_momentum, epsilon, id + ":SGD_momentum", ref errors);
            equals &= Utils.Equals(SGD_usenesterov, other.SGD_usenesterov, id + ":SGD_usenesterov", ref errors);

            equals &= Utils.Equals((int)LearningRateSchedulerType, (int)other.LearningRateSchedulerType, id + ":LearningRateSchedulerType", ref errors);
            equals &= Utils.Equals(CyclicCosineAnnealing_nbEpochsInFirstRun, other.CyclicCosineAnnealing_nbEpochsInFirstRun, epsilon, id + ":CyclicCosineAnnealing_nbEpochsInFirstRun", ref errors);
            equals &= Utils.Equals(CyclicCosineAnnealing_nbEpochInNextRunMultiplier, other.CyclicCosineAnnealing_nbEpochInNextRunMultiplier, epsilon, id + ":CyclicCosineAnnealing_nbEpochInNextRunMultiplier", ref errors);
            equals &= Utils.Equals(OneCycle_DividerForMinLearningRate, other.OneCycle_DividerForMinLearningRate, id + ":OneCycle_DividerForMinLearningRate", ref errors);
            equals &= Utils.Equals(OneCycle_PercentInAnnealing, other.OneCycle_PercentInAnnealing, epsilon, id + ":OneCycle_PercentInAnnealing", ref errors);
            equals &= Utils.Equals(DisableReduceLROnPlateau, other.DisableReduceLROnPlateau, id + ":DisableReduceLROnPlateau", ref errors);
            equals &= Utils.Equals(DivideBy10OnPlateau, other.DivideBy10OnPlateau, id + ":DivideBy10OnPlateau", ref errors);
            equals &= Utils.Equals(LinearLearningRate, other.LinearLearningRate, id + ":LinearLearningRate", ref errors);

            equals &= Utils.Equals(lambdaL2Regularization, other.lambdaL2Regularization, epsilon, id + ":lambdaL2Regularization", ref errors);
            equals &= Utils.Equals(MinimumLearningRate, other.MinimumLearningRate, epsilon, id + ":MinimumLearningRate", ref errors);
            equals &= Utils.Equals(UseDoublePrecision, other.UseDoublePrecision, id + ":UseDoublePrecision", ref errors);
            equals &= Utils.Equals(ForceTensorflowCompatibilityMode, other.ForceTensorflowCompatibilityMode, id + ":ForceTensorflowCompatibilityMode", ref errors);
            equals &= Utils.Equals(DisplayTensorContentStats, other.DisplayTensorContentStats, id + ":DisplayTensorContentStats", ref errors);
            equals &= Utils.Equals(ProfileApplication, other.ProfileApplication, id + ":ProfileApplication", ref errors);
            equals &= Utils.Equals(AutoSaveIntervalInMinuts, other.AutoSaveIntervalInMinuts, id + ":AutoSaveIntervalInMinuts", ref errors);
            equals &= Utils.Equals(SaveNetworkStatsAfterEachEpoch, other.SaveNetworkStatsAfterEachEpoch, id + ":SaveNetworkStatsAfterEachEpoch", ref errors);
            equals &= Utils.Equals(SaveLossAfterEachMiniBatch, other.SaveLossAfterEachMiniBatch, id + ":SaveLossAfterEachMiniBatch", ref errors);
            return equals;
        }


        #region Learning Rate Scheduler
        public NetworkConfig WithCyclicCosineAnnealingLearningRateScheduler(int nbEpochsInFirstRun, int nbEpochInNextRunMultiplier)
        {
            DisableReduceLROnPlateau = true;
            LearningRateSchedulerType = LearningRateSchedulerEnum.CyclicCosineAnnealing;
            CyclicCosineAnnealing_nbEpochsInFirstRun = nbEpochsInFirstRun;
            CyclicCosineAnnealing_nbEpochInNextRunMultiplier = nbEpochInNextRunMultiplier;
            return this;
        }
        public NetworkConfig WithOneCycleLearningRateScheduler(int dividerForMinLearningRate, double percentInAnnealing)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.OneCycle;
            DisableReduceLROnPlateau = true;
            OneCycle_DividerForMinLearningRate = dividerForMinLearningRate;
            OneCycle_PercentInAnnealing = percentInAnnealing;
            return this;
        }
        public NetworkConfig WithCifar10ResNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10ResNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }
        public NetworkConfig WithCifar10WideResNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10WideResNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }
        public NetworkConfig WithCifar10DenseNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10DenseNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }
        public ILearningRateScheduler GetLearningRateScheduler(double initialLearningRate, int numEpochs)
        {
            switch (LearningRateSchedulerType)
            {
                case LearningRateSchedulerEnum.OneCycle:
                    return new OneCycleLearningRateScheduler(initialLearningRate, OneCycle_DividerForMinLearningRate, OneCycle_PercentInAnnealing, numEpochs);
                case LearningRateSchedulerEnum.CyclicCosineAnnealing:
                    return new CyclicCosineAnnealingLearningRateScheduler(initialLearningRate, CyclicCosineAnnealing_nbEpochsInFirstRun, CyclicCosineAnnealing_nbEpochInNextRunMultiplier, numEpochs);
                case LearningRateSchedulerEnum.Cifar10DenseNet:
                    return LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 150, initialLearningRate / 10, 225, initialLearningRate / 100);
                case LearningRateSchedulerEnum.Cifar10ResNet:
                    return LinearLearningRate
                        ? LearningRateScheduler.InterpolateByInterval(1, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100, 200, initialLearningRate / 100)
                        : LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100, 200, initialLearningRate / 100);
                case LearningRateSchedulerEnum.Cifar10WideResNet:
                    return LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 60, initialLearningRate/5, 120, initialLearningRate/25, 160, initialLearningRate/125);
                default:
                    throw new Exception("unknown LearningRateSchedulerType: "+ LearningRateSchedulerType);
            }
        }

        public ReduceLROnPlateau ReduceLROnPlateau()
        {
            if (DisableReduceLROnPlateau)
            {
                return null;
            }
            var factorForReduceLrOnPlateau = DivideBy10OnPlateau ? 0.1 : Math.Sqrt(0.1);
            return new ReduceLROnPlateau(factorForReduceLrOnPlateau, 5, 5);
        }
        #endregion

        #region serialization
        //TODO add tests
        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(LossFunction), (int)LossFunction).Add(nameof(OptimizerType), (int)OptimizerType)
                .Add(nameof(Adam_beta1), Adam_beta1).Add(nameof(Adam_beta2), Adam_beta2).Add(nameof(Adam_epsilon), Adam_epsilon)
                .Add(nameof(SGD_momentum), SGD_momentum).Add(nameof(SGD_usenesterov), SGD_usenesterov)

                //learning rate scheduler fields
                .Add(nameof(LearningRateSchedulerType), (int)LearningRateSchedulerType)
                .Add(nameof(CyclicCosineAnnealing_nbEpochsInFirstRun), CyclicCosineAnnealing_nbEpochsInFirstRun).Add(nameof(CyclicCosineAnnealing_nbEpochInNextRunMultiplier), CyclicCosineAnnealing_nbEpochInNextRunMultiplier)
                .Add(nameof(OneCycle_DividerForMinLearningRate), OneCycle_DividerForMinLearningRate).Add(nameof(OneCycle_PercentInAnnealing), OneCycle_PercentInAnnealing)
                .Add(nameof(DisableReduceLROnPlateau), DisableReduceLROnPlateau).Add(nameof(DivideBy10OnPlateau), DivideBy10OnPlateau).Add(nameof(LinearLearningRate), LinearLearningRate)
                .Add(nameof(lambdaL2Regularization), lambdaL2Regularization)
                .Add(nameof(UseDoublePrecision), UseDoublePrecision)
                .Add(nameof(RandomizeOrder), RandomizeOrder)
                .Add(nameof(ForceTensorflowCompatibilityMode), ForceTensorflowCompatibilityMode)
                .Add(nameof(DisplayTensorContentStats), DisplayTensorContentStats)
                .Add(nameof(ProfileApplication), ProfileApplication)
                .Add(nameof(LogDirectory), LogDirectory)
                .Add(nameof(AutoSaveIntervalInMinuts), AutoSaveIntervalInMinuts)
                .Add(nameof(SaveNetworkStatsAfterEachEpoch), SaveNetworkStatsAfterEachEpoch)
                .Add(nameof(SaveLossAfterEachMiniBatch), SaveLossAfterEachMiniBatch)
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

            //optimizers fields
            OptimizerType = (Optimizer.OptimizationEnum)serialized[nameof(OptimizerType)];
            Adam_beta1 = (double)serialized[nameof(Adam_beta1)];
            Adam_beta2 = (double)serialized[nameof(Adam_beta2)];
            Adam_epsilon = (double)serialized[nameof(Adam_epsilon)];
            SGD_momentum= (double)serialized[nameof(SGD_momentum)];
            SGD_usenesterov = (bool)serialized[nameof(SGD_usenesterov)];
            lambdaL2Regularization = (double)serialized[nameof(lambdaL2Regularization)];

            //learning rate scheduler fields
            LearningRateSchedulerType = (LearningRateSchedulerEnum)serialized[nameof(LearningRateSchedulerType)];
            CyclicCosineAnnealing_nbEpochsInFirstRun = (int)serialized[nameof(CyclicCosineAnnealing_nbEpochsInFirstRun)];
            CyclicCosineAnnealing_nbEpochInNextRunMultiplier = (int)serialized[nameof(CyclicCosineAnnealing_nbEpochInNextRunMultiplier)];
            OneCycle_DividerForMinLearningRate = (int)serialized[nameof(OneCycle_DividerForMinLearningRate)];
            OneCycle_PercentInAnnealing = (double)serialized[nameof(OneCycle_PercentInAnnealing)];
            DisableReduceLROnPlateau = (bool)serialized[nameof(DisableReduceLROnPlateau)];
            DivideBy10OnPlateau = (bool)serialized[nameof(DivideBy10OnPlateau)];
            LinearLearningRate = (bool)serialized[nameof(LinearLearningRate)];
            UseDoublePrecision = (bool)serialized[nameof(UseDoublePrecision)];

            RandomizeOrder = (bool)serialized[nameof(RandomizeOrder)];
            ForceTensorflowCompatibilityMode = (bool)serialized[nameof(ForceTensorflowCompatibilityMode)];
            DisplayTensorContentStats = (bool)serialized[nameof(DisplayTensorContentStats)];
            ProfileApplication = (bool)serialized[nameof(ProfileApplication)];
            LogDirectory = (string)serialized[nameof(LogDirectory)];
            AutoSaveIntervalInMinuts = (int)serialized[nameof(AutoSaveIntervalInMinuts)];
            SaveNetworkStatsAfterEachEpoch = (bool)serialized[nameof(SaveNetworkStatsAfterEachEpoch)];
            SaveLossAfterEachMiniBatch = (bool)serialized[nameof(SaveLossAfterEachMiniBatch)];
            MinimumLearningRate = (double)serialized[nameof(MinimumLearningRate)];
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
