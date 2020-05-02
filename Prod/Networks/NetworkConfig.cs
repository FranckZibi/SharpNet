using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Optimizers;
using static SharpNet.GPU.GPUWrapper;
// ReSharper disable UnusedMember.Global
// ReSharper disable AutoPropertyCanBeMadeGetOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks
{
    public class NetworkConfig
    {
        #region fields

        #region learning rate scheduler fields
        public enum LearningRateSchedulerEnum { Cifar10ResNet, Cifar10DenseNet, OneCycle, CyclicCosineAnnealing, Cifar10WideResNet}

        private LearningRateSchedulerEnum LearningRateSchedulerType { get; set; } = LearningRateSchedulerEnum.Cifar10ResNet;
        private int CyclicCosineAnnealing_nbEpochsInFirstRun { get; set; } = 10;

        private int CyclicCosineAnnealing_nbEpochInNextRunMultiplier { get; set; } = 2;
        //for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        private int OneCycle_DividerForMinLearningRate { get; set; } = 10;
        private double OneCycle_PercentInAnnealing { get; set; } = 0.2;
        public bool DisableReduceLROnPlateau { get; set; }
        private bool DivideBy10OnPlateau { get; set; } = true; // 'true' : validated on 19-apr-2019: +20 bps
        private bool LinearLearningRate { get; set; }
        #endregion

        public LossFunctionEnum LossFunction { get; set;} = LossFunctionEnum.CategoricalCrossentropy;
        public CompatibilityModeEnum CompatibilityMode { get; set;} = CompatibilityModeEnum.SharpNet;

        /// <summary>
        /// The convolution algo to be used
        /// </summary>
        public ConvolutionAlgoPreference ConvolutionAlgoPreference { get; set;} = ConvolutionAlgoPreference.FASTEST_DETERMINIST;

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

        public string DataSetName { get; set; }

        public DataAugmentationConfig DataAugmentation { get; set; }


        public bool RandomizeOrder { get; set; } = true;

        /// <summary>
        /// true if we should use the same conventions then TensorFlow
        /// </summary>
        public bool TensorFlowCompatibilityMode => CompatibilityMode == CompatibilityModeEnum.TensorFlow1 || CompatibilityMode == CompatibilityModeEnum.TensorFlow2;
        /// <summary>
        /// true if we want t display statistics about the weights tensors.
        /// Used only for debugging 
        /// </summary>
        public bool DisplayTensorContentStats{ get; set; }

        /// <summary>
        /// Interval in minutes for saving the network
        /// If less then 0
        ///     => this option will be disabled
        /// If == 0
        ///     => the network will be saved after each iteration
        /// </summary>
        public int AutoSaveIntervalInMinutes { get; set; } = 5*60;
        public bool SaveNetworkStatsAfterEachEpoch { get; set; }

        /// <summary>
        /// name of the the first layer for which we want ot freeze the weights
        /// if 'FirstLayerNameToFreeze' is valid and 'LastLayerNameToFreeze' is empty
        ///     we'll freeze all layers in the network from 'FirstLayerNameToFreeze' to the last network layer
        /// if both 'FirstLayerNameToFreeze' and 'LastLayerNameToFreeze' are valid
        ///     we'll freeze all layers in the network between 'FirstLayerNameToFreeze' and 'LastLayerNameToFreeze'
        /// if 'FirstLayerNameToFreeze' is empty and 'LastLayerNameToFreeze' is valid
        ///     we'll freeze all layers from the start of the network to layer 'LastLayerNameToFreeze'
        /// if both 'FirstLayerNameToFreeze' and 'LastLayerNameToFreeze' are empty
        ///     no layers will be freezed
        /// </summary>
        public string FirstLayerNameToFreeze { get; set; } = "";
        /// <summary>
        /// name of the the last layer for which we want to freeze the weights
        /// </summary>
        public string LastLayerNameToFreeze { get; set; } = "";

        public string LogDirectory { get; set; } = DefaultLogDirectory;
        public static string DefaultLogDirectory => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpNet");

        public static string DefaultDataDirectory => Path.Combine(DefaultLogDirectory, "Data");
        #endregion

        public NetworkConfig()
        {
            Rand = new Random(0);
            DataAugmentation = new DataAugmentationConfig();
        }

        public bool DisableLogging => ReferenceEquals(Logger, Logger.NullLogger);

        // ReSharper disable once MemberCanBeMadeStatic.Global
        public int TypeSize => 4;
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
            equals &= Utils.Equals(LossFunction, other.LossFunction, id + nameof(LossFunction), ref errors);
            equals &= Utils.Equals(Adam_beta1, other.Adam_beta1, epsilon, id + nameof(Adam_beta1), ref errors);
            equals &= Utils.Equals(Adam_beta2, other.Adam_beta2, epsilon, id + nameof(Adam_beta2), ref errors);
            equals &= Utils.Equals(Adam_epsilon, other.Adam_epsilon, epsilon, id + nameof(Adam_epsilon), ref errors);
            equals &= Utils.Equals(SGD_momentum, other.SGD_momentum, epsilon, id + nameof(SGD_momentum), ref errors);
            equals &= Utils.Equals(SGD_usenesterov, other.SGD_usenesterov, id + nameof(SGD_usenesterov), ref errors);
            equals &= Utils.Equals((int)LearningRateSchedulerType, (int)other.LearningRateSchedulerType, id + nameof(LearningRateSchedulerType), ref errors);
            equals &= Utils.Equals(CyclicCosineAnnealing_nbEpochsInFirstRun, other.CyclicCosineAnnealing_nbEpochsInFirstRun, epsilon, id + nameof(CyclicCosineAnnealing_nbEpochsInFirstRun), ref errors);
            equals &= Utils.Equals(CyclicCosineAnnealing_nbEpochInNextRunMultiplier, other.CyclicCosineAnnealing_nbEpochInNextRunMultiplier, epsilon, id + nameof(CyclicCosineAnnealing_nbEpochInNextRunMultiplier), ref errors);
            equals &= Utils.Equals(OneCycle_DividerForMinLearningRate, other.OneCycle_DividerForMinLearningRate, id + nameof(OneCycle_DividerForMinLearningRate), ref errors);
            equals &= Utils.Equals(OneCycle_PercentInAnnealing, other.OneCycle_PercentInAnnealing, epsilon, id + nameof(OneCycle_PercentInAnnealing), ref errors);
            equals &= Utils.Equals(DisableReduceLROnPlateau, other.DisableReduceLROnPlateau, id + nameof(DisableReduceLROnPlateau), ref errors);
            equals &= Utils.Equals(DivideBy10OnPlateau, other.DivideBy10OnPlateau, id + nameof(DivideBy10OnPlateau), ref errors);
            equals &= Utils.Equals(LinearLearningRate, other.LinearLearningRate, id + nameof(LinearLearningRate), ref errors);
            equals &= Utils.Equals(lambdaL2Regularization, other.lambdaL2Regularization, epsilon, id + nameof(lambdaL2Regularization), ref errors);
            equals &= Utils.Equals(MinimumLearningRate, other.MinimumLearningRate, epsilon, id + nameof(MinimumLearningRate), ref errors);
            equals &= Utils.Equals(CompatibilityMode, other.CompatibilityMode, id + nameof(CompatibilityMode), ref errors);
            equals &= Utils.Equals(ConvolutionAlgoPreference, other.ConvolutionAlgoPreference, id + nameof(ConvolutionAlgoPreference), ref errors);
            equals &= Utils.Equals(DisplayTensorContentStats, other.DisplayTensorContentStats, id + nameof(DisplayTensorContentStats), ref errors);
            equals &= Utils.Equals(AutoSaveIntervalInMinutes, other.AutoSaveIntervalInMinutes, id + nameof(AutoSaveIntervalInMinutes), ref errors);
            equals &= Utils.Equals(SaveNetworkStatsAfterEachEpoch, other.SaveNetworkStatsAfterEachEpoch, id + nameof(SaveNetworkStatsAfterEachEpoch), ref errors);
            equals &= DataAugmentation.Equals(other.DataAugmentation, epsilon, id + nameof(DataAugmentation), ref errors);
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
        public ILearningRateComputer GetLearningRateComputer(double initialLearningRate, int epochCount)
        {
            var learningRateScheduler = GetLearningRateScheduler(initialLearningRate, epochCount);
            return new LearningRateComputer(learningRateScheduler, ReduceLROnPlateau(), MinimumLearningRate);
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
        private ILearningRateScheduler GetLearningRateScheduler(double initialLearningRate, int numEpochs)
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
                    return LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 60, initialLearningRate / 5, 120, initialLearningRate / 25, 160, initialLearningRate / 125);
                default:
                    throw new Exception("unknown LearningRateSchedulerType: " + LearningRateSchedulerType);
            }
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
                .Add(nameof(RandomizeOrder), RandomizeOrder)
                .Add(nameof(CompatibilityMode), (int)CompatibilityMode)
                .Add(nameof(ConvolutionAlgoPreference), (int)ConvolutionAlgoPreference)
                .Add(nameof(DisplayTensorContentStats), DisplayTensorContentStats)
                .Add(nameof(LogDirectory), LogDirectory)
                .Add(nameof(AutoSaveIntervalInMinutes), AutoSaveIntervalInMinutes)
                .Add(nameof(SaveNetworkStatsAfterEachEpoch), SaveNetworkStatsAfterEachEpoch)
                .Add(nameof(MinimumLearningRate), MinimumLearningRate)
                .Add(Logger.Serialize())
                .Add(DataAugmentation.Serialize())
                .ToString();
        }
        public static NetworkConfig ValueOf(IDictionary<string, object> serialized)
        {
            return new NetworkConfig(serialized);
        }

        public NetworkConfig Clone()
        {
            return new NetworkConfig(Serializer.Deserialize(Serialize(), null));
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
            RandomizeOrder = (bool)serialized[nameof(RandomizeOrder)];
            CompatibilityMode = (CompatibilityModeEnum)serialized[nameof(CompatibilityMode)];
            ConvolutionAlgoPreference = (ConvolutionAlgoPreference)serialized[nameof(ConvolutionAlgoPreference)];
            DisplayTensorContentStats = (bool)serialized[nameof(DisplayTensorContentStats)];
            LogDirectory = (string)serialized[nameof(LogDirectory)];
            AutoSaveIntervalInMinutes =
                serialized.ContainsKey(nameof(AutoSaveIntervalInMinutes))
                    ? (int) serialized[nameof(AutoSaveIntervalInMinutes)]
                    : (int) serialized["AutoSaveIntervalInMinuts"];
            SaveNetworkStatsAfterEachEpoch = (bool)serialized[nameof(SaveNetworkStatsAfterEachEpoch)];
            MinimumLearningRate = (double)serialized[nameof(MinimumLearningRate)];
            Logger = Logger.ValueOf(serialized);
            DataAugmentation = DataAugmentationConfig.ValueOf(serialized);
            Rand = new Random(0);
        }
        #endregion

        public enum CompatibilityModeEnum
        {
            SharpNet,
            TensorFlow1, //TensorFlow v1
            TensorFlow2, //TensorFlow v2
            PyTorch,
            Caffe,
            MXNet
        }

        /// <summary>
        /// Conventions: 
        ///   the output layer has a shape of (N, C) where:
        ///       'N' is the number of batches
        ///       'C' the number of distinct categoryCount
        /// </summary>
        public enum LossFunctionEnum
        {
            /// <summary>
            /// To be used with sigmoid activation layer.
            /// In a single row, each value will be in [0,1] range
            /// Support of multi labels (one element can belong to several categoryCount at the same time)
            /// </summary>
            BinaryCrossentropy,

            /// <summary>
            /// To be used with softmax activation layer.
            /// In a single row, each value will be in [0,1] range, and the sum of all values wil be equal to 1.0 (= 100%)
            /// Do not support multi labels (each element can belong to exactly 1 category)
            /// </summary>
            CategoricalCrossentropy
        }
    }
}
