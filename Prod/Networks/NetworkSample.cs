using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;
using SharpNet.Optimizers;
using static SharpNet.GPU.GPUWrapper;
// ReSharper disable UnusedMember.Global
// ReSharper disable AutoPropertyCanBeMadeGetOnly.Global
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable ConvertToConstant.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Networks
{
    public class NetworkSample : AbstractModelSample
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
        public bool ShuffleDatasetBeforeEachEpoch = true;
        //when ShuffleDatasetBeforeEachEpoch is true, consider that the dataset is built from block of 'ShuffleDatasetBeforeEachEpochBlockSize' element (that must be kept in same order)
        public int ShuffleDatasetBeforeEachEpochBlockSize = 1;
        public Optimizer.OptimizationEnum OptimizerType = Optimizer.OptimizationEnum.VanillaSGD;


        #region debugging options

        /// <summary>
        /// when set to true, will log all forward and backward propagation
        /// </summary>
        public bool LogNetworkPropagation { get; set; } = false;
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


        public double MinimumRankingScoreToSaveModel = double.NaN;


        /// <summary>
        /// if set:
        ///  we'll only save the model after the epoch that gives the better results:
        ///     better ranking score in validation dataset (if a validation dataset is provided)
        ///     better ranking score in training dataset (if no validation dataset is provided)
        /// </summary>
        public bool use_best_model = true;

        public override IScore GetMinimumRankingScoreToSaveModel()
        {
            if (double.IsNaN(MinimumRankingScoreToSaveModel))
            {
                return null;
            }
            return new Score((float)MinimumRankingScoreToSaveModel, GetRankingEvaluationMetric());

        }


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
        public override EvaluationMetricEnum GetLoss() => LossFunction;
        public override EvaluationMetricEnum GetRankingEvaluationMetric()
        {
            var metrics = GetAllEvaluationMetrics();
            return metrics.Count != 0 ? metrics[0] : EvaluationMetricEnum.DEFAULT_VALUE;
        }

        protected override List<EvaluationMetricEnum> GetAllEvaluationMetrics()
        {
            return EvaluationMetrics;
        }
        public EvaluationMetricEnum LossFunction = EvaluationMetricEnum.DEFAULT_VALUE;


        public float MseOfLog_Epsilon = 0.0008f;
        public float Huber_Delta = 1.0f;

        /// <summary>
        /// the percent of elements in the True (y==1) class
        /// the goal is to recalibrate the loss if one class (y==1 or y==0) is over-represented
        /// </summary>
        public float BCEWithFocalLoss_PercentageInTrueClass = 0.5f;
        public float BCEWithFocalLoss_Gamma = 0;

        public List<EvaluationMetricEnum> EvaluationMetrics = new();

        public CompatibilityModeEnum CompatibilityMode = CompatibilityModeEnum.SharpNet;
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

        public override void Use_All_Available_Cores()
        {
            if (GetDeviceCount() == 0)
            {
                SetResourceId(-1); //we'll use all available CPU
            }
            else
            {
                SetResourceId(int.MaxValue); //we'll use all available GPU
            }
        }


        public override bool FixErrors()
        {
            switch (OptimizerType)
            {
                case Optimizer.OptimizationEnum.AdamW:
                    WithAdamW(AdamW_L2Regularization, Adam_beta1, Adam_beta2, Adam_epsilon);
                    //lambdaL2Regularization = SGD_momentum = 0;
                    //SGD_usenesterov = false;
                    break;
                case Optimizer.OptimizationEnum.SGD:
                    WithSGD(SGD_momentum, SGD_usenesterov);
                    //AdamW_L2Regularization = Adam_beta1 = Adam_beta2 = Adam_epsilon = 0;
                    break;
                case Optimizer.OptimizationEnum.Adam:
                    WithAdam(Adam_beta1, Adam_beta2, Adam_epsilon);
                    //AdamW_L2Regularization = SGD_momentum = 0;
                    //SGD_usenesterov = false;
                    break;
                case Optimizer.OptimizationEnum.VanillaSGD:
                case Optimizer.OptimizationEnum.VanillaSGDOrtho:
                    //SGD_momentum = AdamW_L2Regularization = Adam_beta1 = Adam_beta2 = Adam_epsilon = 0;
                    //SGD_usenesterov = false;
                    break; // no extra configuration needed
            }

            switch (LearningRateSchedulerType)
            {
                case LearningRateSchedulerEnum.CyclicCosineAnnealing:
                    WithCyclicCosineAnnealingLearningRateScheduler(CyclicCosineAnnealing_nbEpochsInFirstRun, CyclicCosineAnnealing_nbEpochInNextRunMultiplier, CyclicCosineAnnealing_MinLearningRate);
                    break;
                case LearningRateSchedulerEnum.OneCycle:
                    WithOneCycleLearningRateScheduler(OneCycle_DividerForMinLearningRate, OneCycle_PercentInAnnealing);
                    break;
                case LearningRateSchedulerEnum.Linear:
                    WithLinearLearningRateScheduler(Linear_DividerForMinLearningRate);
                    break;
            }

            if (!MustUseGPU)
            {
                //this is the only supported mode on CPU
                ConvolutionAlgoPreference = ConvolutionAlgoPreference.FASTEST_DETERMINIST;
            }

            if (DataAugmentationType == ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION)
            {
                ZoomRange = RotationRangeInDegrees = AlphaCutMix = AlphaMixup = CutoutPatchPercentage = RowsCutoutPatchPercentage = ColumnsCutoutPatchPercentage = FillModeConstantVal = AlphaMixup = AlphaCutMix = WidthShiftRangeInPercentage = HeightShiftRangeInPercentage = 0;
                HorizontalFlip = VerticalFlip = Rotate180Degrees= Rotate90Degrees = false;
            }

            if (AlphaMixup>0 && AlphaCutMix>0)
            {
                // Mixup and CutMix can not be used at the same time: we need to disable one of them
                if (Utils.RandomCoinFlip())
                {
                    AlphaMixup = 0; //We disable Mixup
                }
                else
                {
                    AlphaCutMix = 0; //We disable CutMix
                }
            }

            if (LossFunction != EvaluationMetricEnum.BCEWithFocalLoss)
            {
                BCEWithFocalLoss_PercentageInTrueClass = 0.5f; //balanced dataset
                BCEWithFocalLoss_Gamma = 0f;
            }

            if (CutoutPatchPercentage <= 0) { CutoutCount = 0; }
            if (CutoutPatchPercentage > 0) { CutoutCount = Math.Max(CutoutCount, 1); }

            if (ColumnsCutoutPatchPercentage <= 0) { ColumnsCutoutCount = 0; }
            if (ColumnsCutoutPatchPercentage > 0) { ColumnsCutoutCount = Math.Max(ColumnsCutoutCount, 1); }

            if (RowsCutoutPatchPercentage <= 0) { RowsCutoutCount = 0; }
            if (RowsCutoutPatchPercentage > 0) { RowsCutoutCount = Math.Max(RowsCutoutCount, 1); }

            if (LossFunction == EvaluationMetricEnum.DEFAULT_VALUE)
            {
                return false;
            }
            if (EvaluationMetrics.Count == 0)
            {
                EvaluationMetrics = new List<EvaluationMetricEnum> { GetLoss() };
            }

            return true;
        }

        // ReSharper disable once MemberCanBeMadeStatic.Global
        public int TypeSize => 4;
        public override bool MustUseGPU => ResourceIds.Max() >= 0;
        public override void SetTaskId(int taskId)
        {
            if (MustUseGPU)
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

        public virtual void BuildLayers(Network nn, AbstractDatasetSample datasetSample)
        {
            ISample.Log.Warn($"{nameof(BuildLayers)} is not overriden and will do nothing");
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



        public NetworkSample WithSGD(double momentum = 0.9, bool useNesterov = true)
        {
            Debug.Assert(momentum >= 0);
            Debug.Assert(momentum <= 1.0);
            SGD_momentum = momentum;
            SGD_usenesterov = useNesterov;
            OptimizerType = Optimizer.OptimizationEnum.SGD;
            return this;
        }

        public Optimizer GetOptimizer(int[] weightShape, int[] biasShape, TensorMemoryPool memoryPool)
        {
            switch (OptimizerType)
            {
                case Optimizer.OptimizationEnum.Adam:
                    if (Math.Abs(AdamW_L2Regularization) > 1e-6)
                    {
                        throw new Exception("Invalid AdamW_L2Regularization (" + AdamW_L2Regularization + ") for Adam: should be 0");
                    }
                    return new Adam(memoryPool, Adam_beta1, Adam_beta2, Adam_epsilon, 0.0, weightShape, biasShape);
                case Optimizer.OptimizationEnum.AdamW:
                    if (Math.Abs(lambdaL2Regularization) > 1e-6)
                    {
                        throw new Exception("Can't use both AdamW and L2 Regularization");
                    }
                    if (AdamW_L2Regularization < 1e-6)
                    {
                        throw new Exception("Invalid AdamW_L2Regularization (" + AdamW_L2Regularization + ") for AdamW: should be > 0");
                    }
                    return new Adam(memoryPool, Adam_beta1, Adam_beta2, Adam_epsilon, AdamW_L2Regularization, weightShape, biasShape);
                case Optimizer.OptimizationEnum.SGD:
                    return new Sgd(memoryPool, SGD_momentum, SGD_usenesterov, weightShape, biasShape);
                case Optimizer.OptimizationEnum.VanillaSGDOrtho:
                    return new VanillaSgdOrtho(memoryPool, weightShape);
                case Optimizer.OptimizationEnum.VanillaSGD:
                default:
                    return VanillaSgd.Instance;
            }
        }


        public enum CompatibilityModeEnum
        {
            SharpNet,
            TensorFlow,
        }

        
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
        /// randomly rotate the image by 90°
        /// </summary>
        public bool Rotate90Degrees = false;

        /// <summary>
        /// set mode for filling points outside the input boundaries
        /// </summary>
        public ImageDataGenerator.FillModeEnum FillMode = ImageDataGenerator.FillModeEnum.Nearest;

        /// <summary>
        ///value used for fill_mode
        /// </summary>
        public double FillModeConstantVal = 0.0;

        /// <summary>
        /// The cutout to use in maximum % of the longest length ( = Max(height, width) )
        /// ( = % of the max(width,height) of the zero mask to apply to the input picture) (see: https://arxiv.org/pdf/1708.04552.pdf)
        /// recommended size : 16/32=0.5 (= 16x16) for CIFAR-10 / 8/32=0.25 (= 8x8) for CIFAR-100 / 20/32 (= 20x20) for SVHN / 32/96 (= 32x32) for STL-10
        /// If less or equal to 0 , Cutout will be disabled
        /// </summary>
        public double CutoutPatchPercentage = 0.0;
        /// <summary>
        /// number of distinct 'cutout' to perform in the same image
        /// if CutoutPatchPercentage is less or equal to 0 , Cutout will be disabled (and CutoutCount will be set to 0
        /// </summary>
        public int CutoutCount = 1;

        /// <summary>
        /// The columns cutout to use in maximum % of the number of columns
        /// If less or equal to 0 , Columns Cutout will be disabled
        /// </summary>
        public double ColumnsCutoutPatchPercentage = 0.0;
        /// <summary>
        /// number of distinct 'column cutout' to perform in the same image
        /// if ColumnsCutoutPatchPercentage is less or equal to 0 , Column Cutout will be disabled (and ColumnsCutoutCount will be set to 0)
        /// </summary>
        public int ColumnsCutoutCount = 1;

        /// <summary>
        /// The rows cutout to use in maximum % of the number of rows
        /// If less or equal to 0 , Rows Cutout will be disabled
        /// </summary>
        public double RowsCutoutPatchPercentage = 0.0;
        /// <summary>
        /// number of distinct 'rows cutout' to perform in the same image
        /// if RowsCutoutPatchPercentage is less or equal to 0 , Rows Cutout will be disabled (and RowsCutoutCount will be set to)
        /// </summary>
        public int RowsCutoutCount = 1;



        #region time series



        // ReSharper disable once UnusedMember.Global
        public string TimeSeriesDescription()
        {
            string res = "_";
            res += "_" + AugmentedFeaturesPercentage.ToString(CultureInfo.InvariantCulture).Replace(".", "_");

            if (UseContinuousFeatureInEachTimeStep)
            {
                res += "_Continuous";
            }
            if (SameAugmentedFeaturesForEachTimeStep)
            {
                res += "_SameFeaturesByTimeStep";
            }
            return res;
        }

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

        public double AlphaRowsCutMix = 0.0;
        public double AlphaColumnsCutMix = 0.0;


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

        public override Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName)
        {
            return new Network(this, datasetSample, workingDirectory, modelName, true);
        }


        #region IMetricConfig interface
        public override float Get_MseOfLog_Epsilon() => MseOfLog_Epsilon;
        public override float Get_Huber_Delta() => Huber_Delta;
        public override float Get_BCEWithFocalLoss_PercentageInTrueClass() => BCEWithFocalLoss_PercentageInTrueClass;
        public override float Get_BCEWithFocalLoss_Gamma() => BCEWithFocalLoss_Gamma;
        #endregion

    }
}
