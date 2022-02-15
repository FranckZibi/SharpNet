using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SharpNet.HyperParameters;
using SharpNet.Optimizers;
using static SharpNet.GPU.GPUWrapper;
// ReSharper disable UnusedMember.Global
// ReSharper disable AutoPropertyCanBeMadeGetOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks
{
    public class NetworkConfig : AbstractSample
    {

        #region constructors
        public NetworkConfig() : base(new HashSet<string>())
        {
        }
        static NetworkConfig()
        {
            Utils.ConfigureGlobalLog4netProperties(DefaultWorkingDirectory, "SharpNet");
        }
        #endregion

        #region Hyper-Parameters
        /// <summary>
        /// The convolution algo to be used
        /// </summary>
        public ConvolutionAlgoPreference ConvolutionAlgoPreference = ConvolutionAlgoPreference.FASTEST_DETERMINIST;

        public double AdamW_L2Regularization;
        public double Adam_beta1;
        public double Adam_beta2;
        public double Adam_epsilon;
        public double SGD_momentum;
        public double lambdaL2Regularization;
        public bool SGD_usenesterov;
        public bool RandomizeOrder = true;
        public Optimizer.OptimizationEnum OptimizerType = Optimizer.OptimizationEnum.VanillaSGD;
        #endregion



        #region Learning Rate Hyper-Parameters

        public double InitialLearningRate;

        /// <summary>
        /// minimum value for the learning rate
        /// </summary>
        public double MinimumLearningRate = 1e-9;

        #region learning rate scheduler fields
        public enum LearningRateSchedulerEnum { Cifar10ResNet, Cifar10DenseNet, OneCycle, CyclicCosineAnnealing, Cifar10WideResNet, Linear, Constant }

        public LearningRateSchedulerEnum LearningRateSchedulerType = LearningRateSchedulerEnum.Constant;
        public int CyclicCosineAnnealing_nbEpochsInFirstRun = 10;

        public int CyclicCosineAnnealing_nbEpochInNextRunMultiplier = 2;
        /// <summary>
        /// for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        /// </summary>
        public int OneCycle_DividerForMinLearningRate = 10;
        public double OneCycle_PercentInAnnealing = 0.2;


        public int Linear_DividerForMinLearningRate = 100;

        /// <summary>
        /// the minimum value for the learning rate (default value:  1e-6)
        /// </summary>
        public double CyclicCosineAnnealing_MinLearningRate = 1e-6;

        public bool DisableReduceLROnPlateau;
        public bool DivideBy10OnPlateau = true; // 'true' : validated on 19-apr-2019: +20 bps
        public bool LinearLearningRate;
        #endregion
        #endregion


        public string ExtraDescription = "";
        /// <summary>
        /// all resources (CPU or GPU) available for the current network
        /// values superior or equal to 0 means GPU resources (device)
        /// values strictly less then 0 mean CPU resources (host)
        /// 
        /// if ResourceIds.Count == 1
        ///     if masterNetworkIfAny == null:
        ///         all computation will be done in a single network (using resource ResourceIds[0])
        ///     else:
        ///         we are in a slave network (using resource ResourceIds[0]) doing part of the parallel computation
        ///         the master network is 'masterNetworkIfAny'.
        /// else: (ResourceIds.Count >= 2)
        ///     we are the master network (using resource ResourceIds[0]) doing part of the parallel computation
        ///     slaves network will use resourceId ResourceIds[1:]
        /// 
        /// for each resourceId in this list:
        ///     if resourceId strictly less then 0:
        ///         use CPU resource (no GPU usage)
        ///     else:
        ///         run the network on the GPU with device Id = resourceId
        /// </summary>
        public List<int> ResourceIds = new() { 0 };
        public void SetResourceId(int resourceId)
        {
            if (resourceId == int.MaxValue)
            {
                //use multi GPU
                ResourceIds = Enumerable.Range(0, GetDeviceCount()).ToList();
            }
            else
            {
                //single resource
                ResourceIds = new List<int> { resourceId };
            }
        }

        public int NumEpochs;
        public int BatchSize;
        public LossFunctionEnum LossFunction = LossFunctionEnum.CategoricalCrossentropy;
        public CompatibilityModeEnum CompatibilityMode = CompatibilityModeEnum.SharpNet;
        public List<MetricEnum> Metrics = new() { MetricEnum.Loss, MetricEnum.Accuracy };
        public string DataSetName;
        /// <summary>
        /// if true
        ///     we'll always use the full test data set to compute the loss and accuracy of this test data set
        /// else
        ///     we'll use the full test data set for some specific epochs (the first, the last, etc.)
        ///     and a small part of this test data set for other epochs:
        ///         IDataSet.PercentageToUseForLossAndAccuracyFastEstimate
        /// </summary>
        public bool AlwaysUseFullTestDataSetForLossAndAccuracy = true;
        /// <summary>
        /// true if we should use the same conventions then TensorFlow
        /// </summary>
        public bool TensorFlowCompatibilityMode => CompatibilityMode == CompatibilityModeEnum.TensorFlow1 || CompatibilityMode == CompatibilityModeEnum.TensorFlow2;

        /// <summary>
        /// true if we want to display statistics about the weights tensors.
        /// Used only for debugging 
        /// </summary>
        public bool DisplayTensorContentStats = false;

        public bool SaveNetworkStatsAfterEachEpoch = false;
        /// <summary>
        /// Interval in minutes for saving the network
        /// If less then 0
        ///     => this option will be disabled
        /// If == 0
        ///     => the network will be saved after each iteration
        /// </summary>
        public int AutoSaveIntervalInMinutes = 3*60;
        /// <summary>
        /// number of consecutive epochs with a degradation of the validation loss to
        /// stop training the network.
        /// A value less or equal then 0 means no early stopping
        /// </summary>
        public int EarlyStoppingRounds = 0;
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
        public string FirstLayerNameToFreeze = "";
        /// <summary>
        /// name of the the last layer for which we want to freeze the weights
        /// </summary>
        public string LastLayerNameToFreeze = "";

        #region logging
        public string WorkingDirectory = DefaultWorkingDirectory;
        public string LogFile = "SharpNet";
        public static string DefaultWorkingDirectory => Path.Combine(Utils.LocalApplicationFolderPath,  "SharpNet");
        public static string DefaultDataDirectory => Path.Combine(DefaultWorkingDirectory, "Data");
        #endregion

        #region Learning Rate Scheduler
        public NetworkConfig WithCyclicCosineAnnealingLearningRateScheduler(int nbEpochsInFirstRun, int nbEpochInNextRunMultiplier, double minLearningRate = 0.0)
        {
            DisableReduceLROnPlateau = true;
            LearningRateSchedulerType = LearningRateSchedulerEnum.CyclicCosineAnnealing;
            CyclicCosineAnnealing_nbEpochsInFirstRun = nbEpochsInFirstRun;
            CyclicCosineAnnealing_nbEpochInNextRunMultiplier = nbEpochInNextRunMultiplier;
            CyclicCosineAnnealing_MinLearningRate = minLearningRate;
            return this;
        }
        public NetworkConfig WithLinearLearningRateScheduler(int dividerForMinLearningRate)
        {
            DisableReduceLROnPlateau = true;
            LearningRateSchedulerType = LearningRateSchedulerEnum.Linear;
            Linear_DividerForMinLearningRate = dividerForMinLearningRate;
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

        public NetworkConfig WithConstantLearningRateScheduler(double learningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Constant;
            DisableReduceLROnPlateau = true;
            LinearLearningRate = false;
            InitialLearningRate = learningRate;
            return this;
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


        public ILearningRateComputer GetLearningRateComputer()
        {
            return new LearningRateComputer(GetLearningRateScheduler(), ReduceLROnPlateau(), MinimumLearningRate);
        }

        private ILearningRateScheduler GetLearningRateScheduler()
        {
            switch (LearningRateSchedulerType)
            {
                case LearningRateSchedulerEnum.OneCycle:
                    return new OneCycleLearningRateScheduler(InitialLearningRate, OneCycle_DividerForMinLearningRate, OneCycle_PercentInAnnealing, NumEpochs);
                case LearningRateSchedulerEnum.CyclicCosineAnnealing:
                    return new CyclicCosineAnnealingLearningRateScheduler(CyclicCosineAnnealing_MinLearningRate, InitialLearningRate, CyclicCosineAnnealing_nbEpochsInFirstRun, CyclicCosineAnnealing_nbEpochInNextRunMultiplier, NumEpochs);
                case LearningRateSchedulerEnum.Cifar10DenseNet:
                    return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 150, InitialLearningRate / 10, 225, InitialLearningRate / 100);
                case LearningRateSchedulerEnum.Cifar10ResNet:
                    return LinearLearningRate
                        ? LearningRateScheduler.InterpolateByInterval(1, InitialLearningRate, 80, InitialLearningRate / 10, 120, InitialLearningRate / 100, 200, InitialLearningRate / 100)
                        : LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 80, InitialLearningRate / 10, 120, InitialLearningRate / 100, 200, InitialLearningRate / 100);
                case LearningRateSchedulerEnum.Cifar10WideResNet:
                    return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 60, InitialLearningRate / 5, 120, InitialLearningRate / 25, 160, InitialLearningRate / 125);
                case LearningRateSchedulerEnum.Linear:
                    return LearningRateScheduler.Linear(InitialLearningRate, NumEpochs, InitialLearningRate / Linear_DividerForMinLearningRate);
                case LearningRateSchedulerEnum.Constant:
                    return LearningRateScheduler.Constant(InitialLearningRate);
                default:
                    throw new Exception("unknown LearningRateSchedulerType: " + LearningRateSchedulerType);
            }
        }
        #endregion

        public override bool PostBuild()
        {
            return true; //TODO
        }

        // ReSharper disable once MemberCanBeMadeStatic.Global
        public int TypeSize => 4;
        public NetworkConfig WithAdam(double _beta1 = 0.9, double _beta2 = 0.999, double _epsilon = 1e-8)
        {
            Debug.Assert(_beta1 >= 0);
            Debug.Assert(_beta1 <= 1.0);
            Debug.Assert(_beta2 >= 0);
            Debug.Assert(_beta2 <= 1.0);
            AdamW_L2Regularization = 0.0;
            Adam_beta1 = _beta1;
            Adam_beta2 = _beta2;
            Adam_epsilon = _epsilon;
            OptimizerType = Optimizer.OptimizationEnum.Adam;
            return this;
        }

        public NetworkConfig WithAdamW(double l2Regularization, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            Debug.Assert(beta1 >= 0);
            Debug.Assert(beta1 <= 1.0);
            Debug.Assert(beta2 >= 0);
            Debug.Assert(beta2 <= 1.0);
            Debug.Assert(l2Regularization>1e-6);
            AdamW_L2Regularization = l2Regularization;
            lambdaL2Regularization = 0; //L2 regularization is not compatible with AdamW
            Adam_beta1 = beta1;
            Adam_beta2 = beta2;
            Adam_epsilon = epsilon;
            OptimizerType = Optimizer.OptimizationEnum.AdamW;
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

        public NetworkConfig Clone()
        {
            throw new NotImplementedException();
        }

        public enum CompatibilityModeEnum
        {
            SharpNet,
            TensorFlow1, //TensorFlow v1
            TensorFlow2, //TensorFlow v2
            PyTorch,
            Caffe,
            MXNet
        }


        public const float Default_MseOfLog_Loss = 0.0008f;

        public static NetworkConfig ValueOf(string workingDirectory, string modelName)
        {
            return (NetworkConfig)ISample.LoadConfigIntoSample(() => new NetworkConfig(), workingDirectory, modelName);
        }
    }
}
