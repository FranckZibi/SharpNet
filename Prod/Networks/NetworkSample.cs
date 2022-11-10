using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.Models;
using SharpNet.Optimizers;
using static SharpNet.GPU.GPUWrapper;
// ReSharper disable UnusedMember.Global
// ReSharper disable AutoPropertyCanBeMadeGetOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks
{
    public class NetworkSample : AbstractSample, IModelSample
    {

        #region constructors
        public NetworkSample() : base(new HashSet<string>())
        {
        }
        static NetworkSample()
        {
            Utils.ConfigureGlobalLog4netProperties(DefaultWorkingDirectory, "SharpNet");
        }
        #endregion

        #region Hyper-Parameters
        /// <summary>
        /// The convolution algo to be used
        /// </summary>
        public ConvolutionAlgoPreference ConvolutionAlgoPreference = ConvolutionAlgoPreference.FASTEST_DETERMINIST;

        public double AdamW_L2Regularization = 0.01;
        public double Adam_beta1 = 0.9;
        public double Adam_beta2 = 0.999;
        public double Adam_epsilon = 1e-8;
        public double SGD_momentum = 0.9;
        public double lambdaL2Regularization;
        public bool SGD_usenesterov = true;
        public bool RandomizeOrder = true;
        //when RandomizeOrder is true, consider that the dataset is built from block of 'RandomizeOrderBlockSize' element (that must be kept in same order)
        public int RandomizeOrderBlockSize = 1;
        public Optimizer.OptimizationEnum OptimizerType = Optimizer.OptimizationEnum.VanillaSGD;
        #endregion



        [SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
        public static NetworkSample ValueOf(string workingDirectory, string modelName)
        {
            try { return ISample.LoadSample<EfficientNetNetworkSample>(workingDirectory, modelName); } catch { }
            try { return ISample.LoadSample<NetworkSample_1DCNN>(workingDirectory, modelName); } catch { }
            try { return ISample.LoadSample<Cfm60NetworkSample>(workingDirectory, modelName); } catch { }
            try { return ISample.LoadSample<WideResNetNetworkSample>(workingDirectory, modelName); } catch { }
            //try { return ValueOfNetworkSample(workingDirectory, modelName); } catch { }
            throw new Exception($"can't load sample from model {modelName} in directory {workingDirectory}");
        }

        

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
        public EvaluationMetricEnum LossFunction = EvaluationMetricEnum.CategoricalCrossentropy;
        public CompatibilityModeEnum CompatibilityMode = CompatibilityModeEnum.SharpNet;
        public List<EvaluationMetricEnum> Metrics = new() { EvaluationMetricEnum.CategoricalCrossentropy, EvaluationMetricEnum.Accuracy };
        public string DataSetName;
        /// <summary>
        /// if true
        ///     we'll always use the full test data set to compute the loss and accuracy of this test data set
        /// else
        ///     we'll use the full test data set for some specific epochs (the first, the last, etc.)
        ///     and a small part of this test data set for other epochs:
        ///         DataSet.PercentageToUseForLossAndAccuracyFastEstimate
        /// </summary>
        public bool AlwaysUseFullTestDataSetForLossAndAccuracy = true;
        /// <summary>
        /// true if we should use the same conventions then TensorFlow
        /// </summary>
        public bool TensorFlowCompatibilityMode => CompatibilityMode == CompatibilityModeEnum.TensorFlow;

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
        public static string DefaultWorkingDirectory => Utils.ChallengesPath;
        public static string DefaultDataDirectory => Path.Combine(DefaultWorkingDirectory, "Data");
        #endregion

        #region Learning Rate Scheduler
        public NetworkSample WithCyclicCosineAnnealingLearningRateScheduler(int nbEpochsInFirstRun, int nbEpochInNextRunMultiplier, double minLearningRate = 0.0)
        {
            DisableReduceLROnPlateau = true;
            LearningRateSchedulerType = LearningRateSchedulerEnum.CyclicCosineAnnealing;
            CyclicCosineAnnealing_nbEpochsInFirstRun = nbEpochsInFirstRun;
            CyclicCosineAnnealing_nbEpochInNextRunMultiplier = nbEpochInNextRunMultiplier;
            CyclicCosineAnnealing_MinLearningRate = minLearningRate;
            return this;
        }
        public NetworkSample WithLinearLearningRateScheduler(int dividerForMinLearningRate)
        {
            DisableReduceLROnPlateau = true;
            LearningRateSchedulerType = LearningRateSchedulerEnum.Linear;
            Linear_DividerForMinLearningRate = dividerForMinLearningRate;
            return this;
        }
        public NetworkSample WithOneCycleLearningRateScheduler(int dividerForMinLearningRate, double percentInAnnealing)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.OneCycle;
            DisableReduceLROnPlateau = true;
            OneCycle_DividerForMinLearningRate = dividerForMinLearningRate;
            OneCycle_PercentInAnnealing = percentInAnnealing;
            return this;
        }
        public NetworkSample WithCifar10ResNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10ResNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }
        public NetworkSample WithCifar10WideResNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10WideResNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }
        public NetworkSample WithCifar10DenseNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10DenseNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }

        public NetworkSample WithConstantLearningRateScheduler(double learningRate)
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


        public override bool FixErrors()
        {
            switch (OptimizerType)
            {
                case Optimizer.OptimizationEnum.AdamW:
                    WithAdamW(AdamW_L2Regularization, Adam_beta1, Adam_beta2, Adam_epsilon);
                    lambdaL2Regularization = SGD_momentum = 0;
                    SGD_usenesterov = false;
                    break;
                case Optimizer.OptimizationEnum.SGD:
                    WithSGD(SGD_momentum, SGD_usenesterov);
                    AdamW_L2Regularization = Adam_beta1 = Adam_beta2 = Adam_epsilon = 0;
                    break;
                case Optimizer.OptimizationEnum.Adam:
                    WithAdam(Adam_beta1, Adam_beta2, Adam_epsilon);
                    AdamW_L2Regularization = SGD_momentum = 0;
                    SGD_usenesterov = false;
                    break;
                case Optimizer.OptimizationEnum.VanillaSGD:
                case Optimizer.OptimizationEnum.VanillaSGDOrtho:
                    SGD_momentum = AdamW_L2Regularization = Adam_beta1 = Adam_beta2 = Adam_epsilon = 0;
                    SGD_usenesterov = false;
                    break; // no extra configuration needed
            }

            switch (LearningRateSchedulerType)
            {
                case LearningRateSchedulerEnum.CyclicCosineAnnealing:
                    WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
                    break;
                case LearningRateSchedulerEnum.OneCycle:
                    WithOneCycleLearningRateScheduler(200, 0.1);
                    break;
                case LearningRateSchedulerEnum.Linear:
                    WithLinearLearningRateScheduler(1000);
                    break;
            }

            if (!UseGPU)
            {
                //this is the only supported mode on CPU
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST;
            }

            return true;
        }

        // ReSharper disable once MemberCanBeMadeStatic.Global
        public int TypeSize => 4;
        public override bool UseGPU => ResourceIds.Max() >= 0;
        public override void SetTaskId(int taskId)
        {
            if (UseGPU)
            {
                ResourceIds = new List<int> { taskId };
            }
            else
            {
                //CPU
                ResourceIds = new List<int> { -1 };
            }
        }

        public NetworkSample WithAdam(double _beta1 = 0.9, double _beta2 = 0.999, double _epsilon = 1e-8)
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="adamW_L2Regularization">also known as weight decay</param>
        /// <param name="beta1"></param>
        /// <param name="beta2"></param>
        /// <param name="epsilon"></param>
        /// <returns></returns>
        public NetworkSample WithAdamW(double adamW_L2Regularization = 0.01, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            Debug.Assert(beta1 >= 0);
            Debug.Assert(beta1 <= 1.0);
            Debug.Assert(beta2 >= 0);
            Debug.Assert(beta2 <= 1.0);
            Debug.Assert(adamW_L2Regularization>1e-6);
            AdamW_L2Regularization = adamW_L2Regularization;
            lambdaL2Regularization = 0; //L2 regularization is not compatible with AdamW
            Adam_beta1 = beta1;
            Adam_beta2 = beta2;
            Adam_epsilon = epsilon;
            OptimizerType = Optimizer.OptimizationEnum.AdamW;
            return this;
        }

        public virtual void BuildLayers(Network network, AbstractDatasetSample datasetSample)
        {
            throw new NotImplementedException();
        }

        public virtual void SaveExtraModelInfos(Model model, string workingDirectory, string modelName)
        {
        }

        public enum POOLING_BEFORE_DENSE_LAYER
        {
            /* we'll use an Average Pooling layer of size [2 x 2] before the Dense layer*/
            AveragePooling_2,
            /* we'll use an Average Pooling layer of size [8 x 8] before the Dense layer*/
            AveragePooling_8,

            /* We'll use a Global Average Pooling (= GAP) layer just before the last Dense (= fully connected) Layer
            This GAP layer will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n,c,1,1),
            so that this output feature map is independent of the the size of the input */
            GlobalAveragePooling,

            /* we'll use a Global Average Pooling layer concatenated with a Global Max Pooling layer before the Dense Layer
            This will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n, 2*c, 1, 1),
            so that this output feature map is independent of the the size of the input */
            GlobalAveragePooling_And_GlobalMaxPooling,

            /* We'll use a Global Max Pooling layer just before the last Dense (= fully connected) Layer
            This will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n,c,1,1),
            so that this output feature map is independent of the the size of the input */
            GlobalMaxPooling,

            /* no pooling */
            NONE
        };

        public Network BuildNetworkWithoutLayers(string workingDirectory, string modelName)
        {
            var network = new Network(this, null, workingDirectory, modelName, false); //!D
            return network;
        }

        //private static NetworkSample ValueOfNetworkSample(string workingDirectory, string modelName)
        //{
        //    return new NetworkSample(new ISample[]
        //    {
        //        ISample.LoadSample<NetworkSample>(workingDirectory, ISample.SampleName(modelName, 0)),
        //        ISample.LoadSample<DataAugmentationSample>(workingDirectory, ISample.SampleName(modelName, 1)),
        //    });
        //}

       

        public EvaluationMetricEnum GetLoss()
        {
            return LossFunction;
        }

        // ReSharper disable once UnusedParameter.Global
        public virtual void ApplyDataset(AbstractDatasetSample datasetSample)
        {
            return;
        }








        public NetworkSample WithSGD(double momentum = 0.9, bool useNesterov = true)
        {
            Debug.Assert(momentum >= 0);
            Debug.Assert(momentum <= 1.0);
            SGD_momentum = momentum;
            SGD_usenesterov = useNesterov;
            OptimizerType = Optimizer.OptimizationEnum.SGD;
            return this;
        }

        public enum CompatibilityModeEnum
        {
            SharpNet,
            TensorFlow,
        }

        public const float Default_MseOfLog_Loss = 0.0008f;

        
        
        #region DataAugmentation

        public ImageDataGenerator.DataAugmentationEnum DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION;
        /// <summary>
        ///randomly shift images horizontally
        /// </summary>
        public double WidthShiftRangeInPercentage = 0.0;
        /// <summary>
        /// randomly shift images vertically
        /// </summary>
        public double HeightShiftRangeInPercentage = 0.0;
        /// <summary>
        /// randomly flip images horizontally
        /// </summary>
        public bool HorizontalFlip = false;
        /// <summary>
        /// randomly flip images vertically
        /// </summary>
        public bool VerticalFlip = false;
        /// <summary>
        /// randomly rotate the image by 180°
        /// </summary>
        public bool Rotate180Degrees = false;

        /// <summary>
        /// set mode for filling points outside the input boundaries
        /// </summary>
        public ImageDataGenerator.FillModeEnum FillMode = ImageDataGenerator.FillModeEnum.Nearest;

        /// <summary>
        ///value used for fill_mode
        /// </summary>
        public double FillModeConstantVal = 0.0;

        /// <summary>
        /// The cutout to use in % of the longest length ( = Max(height, width) )
        /// ( = % of the max(width,height) of the zero mask to apply to the input picture) (see: https://arxiv.org/pdf/1708.04552.pdf)
        /// recommended size : 16/32=0.5 (= 16x16) for CIFAR-10 / 8/32=0.25 (= 8x8) for CIFAR-100 / 20/32 (= 20x20) for SVHN / 32/96 (= 32x32) for STL-10
        /// If less or equal to 0 , Cutout will be disabled
        /// </summary>
        public double CutoutPatchPercentage = 0.0;

        #region time series

        // ReSharper disable once UnusedMember.Global
        public void WithTimeSeriesDataAugmentation(TimeSeriesDataAugmentationEnum timeSeriesDataAugmentationType,
            double augmentedFeaturesPercentage,
            bool useContinuousFeatureInEachTimeStep,
            bool sameAugmentedFeaturesForEachTimeStep,
            double noiseInPercentageOfVolatility = 0.0)
        {
            Debug.Assert(augmentedFeaturesPercentage >= 0);
            Debug.Assert(augmentedFeaturesPercentage <= (1f + 1e-6));
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.TIME_SERIES;
            TimeSeriesDataAugmentationType = timeSeriesDataAugmentationType;
            AugmentedFeaturesPercentage = augmentedFeaturesPercentage;
            UseContinuousFeatureInEachTimeStep = useContinuousFeatureInEachTimeStep;
            SameAugmentedFeaturesForEachTimeStep = sameAugmentedFeaturesForEachTimeStep;
            NoiseInPercentageOfVolatility = noiseInPercentageOfVolatility;
        }

        public enum TimeSeriesDataAugmentationEnum
        {
            NOTHING,         //no change
            REPLACE_BY_MEAN, //replace feature by its mean
            REPLACE_BY_ZERO, //replace feature by zero
            ADD_NOISE,       //add noise to feature
        }


        // ReSharper disable once UnusedMember.Global
        public string TimeSeriesDescription()
        {
            string res = "_";
            res += TimeSeriesDataAugmentationType;
            res += "_" + AugmentedFeaturesPercentage.ToString(CultureInfo.InvariantCulture).Replace(".", "_");

            if (UseContinuousFeatureInEachTimeStep)
            {
                res += "_Continuous";
            }
            if (SameAugmentedFeaturesForEachTimeStep)
            {
                res += "_SameFeaturesByTimeStep";
            }
            if (TimeSeriesDataAugmentationType == TimeSeriesDataAugmentationEnum.ADD_NOISE)
            {
                res += "_noise_" + NoiseInPercentageOfVolatility.ToString(CultureInfo.InvariantCulture).Replace(".", "_");
            }
            return res;
        }

        public TimeSeriesDataAugmentationEnum TimeSeriesDataAugmentationType = TimeSeriesDataAugmentationEnum.NOTHING;
        public bool UseContinuousFeatureInEachTimeStep = false;

        /// <summary>
        /// % of the number of features to be 'augmented'
        /// Ex: 0.2 means 20% of the features will be 'augmented'
        /// </summary>
        public double AugmentedFeaturesPercentage = 0.03;

        public bool SameAugmentedFeaturesForEachTimeStep = false;

        /// <summary>
        /// When TimeSeriesType = TimeSeriesAugmentationType.ADD_NOISE
        /// the % of noise to add to the feature in % of the feature volatility
        /// </summary>
        public double NoiseInPercentageOfVolatility = 0.1;
        #endregion

        /// <summary>
        /// The alpha coefficient used to compute lambda in CutMix
        /// If less or equal to 0 , CutMix will be disabled
        /// Alpha will be used as an input of the beta law to compute lambda
        /// (so a value of AlphaCutMix = 1.0 will use a uniform random distribution in [0,1] for lambda)
        /// lambda is the % of the original to keep (1-lambda will be taken from another element and mixed with current)
        /// the % of the max(width,height) of the CutMix mask to apply to the input picture (see: https://arxiv.org/pdf/1905.04899.pdf)
        /// </summary>
        public double AlphaCutMix = 0.0;

        /// <summary>
        /// The alpha coefficient used to compute lambda in Mixup
        /// A value less or equal to 0.0 wil disable Mixup (see: https://arxiv.org/pdf/1710.09412.pdf)
        /// A value of 1.0 will use a uniform random distribution in [0,1] for lambda
        /// </summary>
        public double AlphaMixup = 0.0;

        /// <summary>
        /// rotation range in degrees, in [0,180] range.
        /// The actual rotation will be a random number in [-_rotationRangeInDegrees,+_rotationRangeInDegrees]
        /// </summary>
        public double RotationRangeInDegrees = 0.0;

        /// <summary>
        /// Range for random zoom. [lower, upper] = [1 - _zoomRange, 1 + _zoomRange].
        /// </summary>
        public double ZoomRange = 0.0;

        /// <summary>
        /// Probability to apply Equalize operation
        /// </summary>
        public double EqualizeOperationProbability = 0.0;

        /// <summary>
        /// Probability to apply AutoContrast operation
        /// </summary>
        public double AutoContrastOperationProbability = 0.0;

        /// <summary>
        /// Probability to apply Invert operation
        /// </summary>
        public double InvertOperationProbability = 0.0;

        /// <summary>
        /// Probability to apply Brightness operation
        /// </summary>
        public double BrightnessOperationProbability = 0.0;
        /// <summary>
        /// The enhancement factor used for Brightness operation (between [0.1,1.9]
        /// </summary>
        public double BrightnessOperationEnhancementFactor = 0.0;

        /// <summary>
        /// Probability to apply Color operation
        /// </summary>
        public double ColorOperationProbability = 0.0;
        /// <summary>
        /// The enhancement factor used for Color operation (between [0.1,1.9]
        /// </summary>
        public double ColorOperationEnhancementFactor = 0.0;

        /// <summary>
        /// Probability to apply Contrast operation
        /// </summary>
        public double ContrastOperationProbability = 0.0;
        /// <summary>
        /// The enhancement factor used for Contrast operation (between [0.1,1.9]
        /// </summary>
        public double ContrastOperationEnhancementFactor = 0.0;

        /// <summary>
        /// The number of operations for the RandAugment
        /// Only used when DataAugmentationType = DataAugmentationEnum.RAND_AUGMENT
        /// </summary>
        public int RandAugment_N = 0;
        /// <summary>
        /// The magnitude of operations for the RandAugment
        /// Only used when DataAugmentationType = DataAugmentationEnum.RAND_AUGMENT
        /// </summary>
        public int RandAugment_M = 0;

        // ReSharper disable once UnusedMember.Global
        public void WithRandAugment(int N, int M)
        {
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.RAND_AUGMENT;
            RandAugment_N = N;
            RandAugment_M = M;
        }
        public bool UseDataAugmentation => DataAugmentationType != ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION;

        //public static DataAugmentationSample ValueOf(string workingDirectory, string sampleName)
        //{
        //    return ISample.LoadSample<DataAugmentationSample>(workingDirectory, sampleName);
        //}

        #endregion


    }
}
