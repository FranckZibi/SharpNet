using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using SharpNet.HyperParameters;
// ReSharper disable AutoPropertyCanBeMadeGetOnly.Local

namespace SharpNet.DataAugmentation
{
    public class DataAugmentationSample : AbstractSample
    {
        public DataAugmentationSample() : base(new HashSet<string>()) { }

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

        public static DataAugmentationSample ValueOf(string workingDirectory, string sampleName)
        {
            return ISample.LoadSample<DataAugmentationSample>(workingDirectory, sampleName);
        }
    }
}
