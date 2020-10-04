using System.Collections.Generic;
using SharpNet.Data;

namespace SharpNet.DataAugmentation
{
    public class DataAugmentationConfig
    {
        public ImageDataGenerator.DataAugmentationEnum DataAugmentationType { get; set; } = ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION;
        /// <summary>
        ///randomly shift images horizontally
        /// </summary>
        public double WidthShiftRangeInPercentage { get; set; } = 0.0;
        /// <summary>
        /// randomly shift images vertically
        /// </summary>
        public double HeightShiftRangeInPercentage { get; set; } = 0.0;
        /// <summary>
        /// randomly flip images horizontally
        /// </summary>
        public bool HorizontalFlip { get; set; } = false;
        /// <summary>
        /// randomly flip images vertically
        /// </summary>
        public bool VerticalFlip { get; set; } = false;
        /// <summary>
        /// randomly rotate the image by 180°
        /// </summary>
        public bool Rotate180Degrees { get; set; } = false;

        /// <summary>
        /// set mode for filling points outside the input boundaries
        /// </summary>
        public ImageDataGenerator.FillModeEnum FillMode { get; set; } = ImageDataGenerator.FillModeEnum.Nearest;

        /// <summary>
        ///value used for fill_mode
        /// </summary>
        public double FillModeConstantVal { get; set; } = 0.0;

        /// <summary>
        /// The cutout to use in % of the longest length ( = Max(height, width) )
        /// ( = % of the max(width,height) of the zero mask to apply to the input picture) (see: https://arxiv.org/pdf/1708.04552.pdf)
        /// recommended size : 16/32=0.5 (= 16x16) for CIFAR-10 / 8/32=0.25 (= 8x8) for CIFAR-100 / 20/32 (= 20x20) for SVHN / 32/96 (= 32x32) for STL-10
        /// If less or equal to 0 , Cutout will be disabled
        /// </summary>
        public double CutoutPatchPercentage { get; set; } = 0.0;

        /// <summary>
        /// The alpha coefficient used to compute lambda in CutMix
        /// If less or equal to 0 , CutMix will be disabled
        /// Alpha will be used as an input of the beta law to compute lambda
        /// (so a value of AlphaCutMix = 1.0 will use a uniform random distribution in [0,1] for lambda)
        /// lambda is the % of the original to keep (1-lambda will be taken from another element and mixed with current)
        /// the % of the max(width,height) of the CutMix mask to apply to the input picture (see: https://arxiv.org/pdf/1905.04899.pdf)
        /// </summary>
        public double AlphaCutMix { get; set; } = 0.0;

        /// <summary>
        /// The alpha coefficient used to compute lambda in Mixup
        /// A value less or equal to 0.0 wil disable Mixup (see: https://arxiv.org/pdf/1710.09412.pdf)
        /// A value of 1.0 will use a uniform random distribution in [0,1] for lambda
        /// </summary>
        public double AlphaMixup { get; set; } = 0.0;

        /// <summary>
        /// rotation range in degrees, in [0,180] range.
        /// The actual rotation will be a random number in [-_rotationRangeInDegrees,+_rotationRangeInDegrees]
        /// </summary>
        public double RotationRangeInDegrees { get; set; } = 0.0;

        /// <summary>
        /// Range for random zoom. [lower, upper] = [1 - _zoomRange, 1 + _zoomRange].
        /// </summary>
        public double ZoomRange { get; set; } = 0.0;

        /// <summary>
        /// Probability to apply Equalize operation
        /// </summary>
        public double EqualizeOperationProbability { get; private set; } = 0.0;

        /// <summary>
        /// Probability to apply AutoContrast operation
        /// </summary>
        public double AutoContrastOperationProbability { get; private set; } = 0.0;

        /// <summary>
        /// Probability to apply Invert operation
        /// </summary>
        public double InvertOperationProbability { get; private set; } = 0.0;

        /// <summary>
        /// Probability to apply Brightness operation
        /// </summary>
        public double BrightnessOperationProbability { get; private set; } = 0.0;
        /// <summary>
        /// The enhancement factor used for Brightness operation (between [0.1,1.9]
        /// </summary>
        public double BrightnessOperationEnhancementFactor { get; private set; } = 0.0;

        /// <summary>
        /// Probability to apply Color operation
        /// </summary>
        public double ColorOperationProbability { get; private set; } = 0.0;
        /// <summary>
        /// The enhancement factor used for Color operation (between [0.1,1.9]
        /// </summary>
        public double ColorOperationEnhancementFactor { get; private set; } = 0.0;

        /// <summary>
        /// Probability to apply Contrast operation
        /// </summary>
        public double ContrastOperationProbability { get; private set; } = 0.0;
        /// <summary>
        /// The enhancement factor used for Contrast operation (between [0.1,1.9]
        /// </summary>
        public double ContrastOperationEnhancementFactor { get; private set; } = 0.0;

        /// <summary>
        /// The number of operations for the RandAugment
        /// Only used when DataAugmentationType = DataAugmentationEnum.RAND_AUGMENT
        /// </summary>
        public int RandAugment_N { get; set; } = 0;
        /// <summary>
        /// The magnitude of operations for the RandAugment
        /// Only used when DataAugmentationType = DataAugmentationEnum.RAND_AUGMENT
        /// </summary>
        public int RandAugment_M { get; set; } = 0;

        // ReSharper disable once UnusedMember.Global
        public void WithRandAugment(int N, int M)
        {
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.RAND_AUGMENT;
            RandAugment_N = N;
            RandAugment_M = M;
        }

        public static readonly DataAugmentationConfig NoDataAugmentation = new DataAugmentationConfig();

        public bool Equals(DataAugmentationConfig other, double epsilon, string id, ref string errors)
        {
            var equals = true;
            equals &= Utils.Equals(DataAugmentationType, other.DataAugmentationType, id + nameof(DataAugmentationType), ref errors);
            equals &= Utils.Equals(WidthShiftRangeInPercentage, other.WidthShiftRangeInPercentage, epsilon, id + nameof(WidthShiftRangeInPercentage), ref errors);
            equals &= Utils.Equals(HeightShiftRangeInPercentage, other.HeightShiftRangeInPercentage, epsilon, id + nameof(HeightShiftRangeInPercentage), ref errors);
            equals &= Utils.Equals(HorizontalFlip, other.HorizontalFlip, id + nameof(HorizontalFlip), ref errors);
            equals &= Utils.Equals(VerticalFlip, other.VerticalFlip, id + nameof(VerticalFlip), ref errors);
            equals &= Utils.Equals(Rotate180Degrees, other.Rotate180Degrees, id + nameof(Rotate180Degrees), ref errors);
            equals &= Utils.Equals(FillMode, other.FillMode, id + nameof(FillMode), ref errors);
            equals &= Utils.Equals(FillModeConstantVal, other.FillModeConstantVal, epsilon, id + nameof(FillModeConstantVal), ref errors);
            equals &= Utils.Equals(CutoutPatchPercentage, other.CutoutPatchPercentage, epsilon, id + nameof(CutoutPatchPercentage), ref errors);
            equals &= Utils.Equals(AlphaCutMix, other.AlphaCutMix, epsilon, id + nameof(AlphaCutMix), ref errors);
            equals &= Utils.Equals(AlphaMixup, other.AlphaMixup, epsilon, id + nameof(AlphaMixup), ref errors);
            equals &= Utils.Equals(RotationRangeInDegrees, other.RotationRangeInDegrees, epsilon, id + nameof(RotationRangeInDegrees), ref errors);
            equals &= Utils.Equals(ZoomRange, other.ZoomRange, epsilon, id + nameof(ZoomRange), ref errors);
            equals &= Utils.Equals(EqualizeOperationProbability, other.EqualizeOperationProbability, epsilon, id + nameof(EqualizeOperationProbability), ref errors);
            equals &= Utils.Equals(AutoContrastOperationProbability, other.AutoContrastOperationProbability, epsilon, id + nameof(AutoContrastOperationProbability), ref errors);
            equals &= Utils.Equals(InvertOperationProbability, other.InvertOperationProbability, epsilon, id + nameof(InvertOperationProbability), ref errors);
            equals &= Utils.Equals(BrightnessOperationProbability, other.BrightnessOperationProbability, epsilon, id + nameof(BrightnessOperationProbability), ref errors);
            equals &= Utils.Equals(BrightnessOperationEnhancementFactor, other.BrightnessOperationEnhancementFactor, epsilon, id + nameof(BrightnessOperationEnhancementFactor), ref errors);
            equals &= Utils.Equals(ColorOperationProbability, other.ColorOperationProbability, epsilon, id + nameof(ColorOperationProbability), ref errors);
            equals &= Utils.Equals(ColorOperationEnhancementFactor, other.ColorOperationEnhancementFactor, epsilon, id + nameof(ColorOperationEnhancementFactor), ref errors);
            equals &= Utils.Equals(ContrastOperationProbability, other.ContrastOperationProbability, epsilon, id + nameof(ContrastOperationProbability), ref errors);
            equals &= Utils.Equals(ContrastOperationEnhancementFactor, other.ContrastOperationEnhancementFactor, epsilon, id + nameof(ContrastOperationEnhancementFactor), ref errors);
            equals &= Utils.Equals(RandAugment_N, other.RandAugment_N, id + nameof(RandAugment_N), ref errors);
            equals &= Utils.Equals(RandAugment_M, other.RandAugment_M, id + nameof(RandAugment_M), ref errors);
            return equals;
        }

        public bool UseDataAugmentation => !ReferenceEquals(this, NoDataAugmentation);

        #region serialization
        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(DataAugmentationType), (int)DataAugmentationType)
                .Add(nameof(WidthShiftRangeInPercentage), WidthShiftRangeInPercentage)
                .Add(nameof(HeightShiftRangeInPercentage), HeightShiftRangeInPercentage)
                .Add(nameof(HorizontalFlip), HorizontalFlip)
                .Add(nameof(VerticalFlip), VerticalFlip)
                .Add(nameof(FillMode), (int)FillMode)
                .Add(nameof(FillModeConstantVal), FillModeConstantVal)
                .Add(nameof(CutoutPatchPercentage), CutoutPatchPercentage)
                .Add(nameof(AlphaCutMix), AlphaCutMix)
                .Add(nameof(AlphaMixup), AlphaMixup)
                .Add(nameof(RotationRangeInDegrees), RotationRangeInDegrees)
                .Add(nameof(ZoomRange), ZoomRange)
                .Add(nameof(EqualizeOperationProbability), EqualizeOperationProbability)
                .Add(nameof(AutoContrastOperationProbability), AutoContrastOperationProbability)
                .Add(nameof(InvertOperationProbability), InvertOperationProbability)
                .Add(nameof(BrightnessOperationProbability), BrightnessOperationProbability)
                .Add(nameof(BrightnessOperationEnhancementFactor), BrightnessOperationEnhancementFactor)
                .Add(nameof(ColorOperationProbability), ColorOperationProbability)
                .Add(nameof(ColorOperationEnhancementFactor), ColorOperationEnhancementFactor)
                .Add(nameof(ContrastOperationProbability), ContrastOperationProbability)
                .Add(nameof(ContrastOperationEnhancementFactor), ContrastOperationEnhancementFactor)
                .Add(nameof(RandAugment_N), RandAugment_N)
                .Add(nameof(RandAugment_M), RandAugment_M)
                .ToString();
        }
        public static DataAugmentationConfig ValueOf(IDictionary<string, object> serialized)
        {
            var da = new DataAugmentationConfig();
            if (!serialized.ContainsKey(nameof(WidthShiftRangeInPercentage)))
            {
                return da;
            }
            da.DataAugmentationType = (ImageDataGenerator.DataAugmentationEnum) serialized[nameof(DataAugmentationType)];
            da.WidthShiftRangeInPercentage = (double) serialized[nameof(WidthShiftRangeInPercentage)];
            da.HeightShiftRangeInPercentage = (double) serialized[nameof(HeightShiftRangeInPercentage)];
            da.HorizontalFlip = (bool) serialized[nameof(HorizontalFlip)];
            da.VerticalFlip = (bool) serialized[nameof(VerticalFlip)];
            da.FillMode = (ImageDataGenerator.FillModeEnum) serialized[nameof(FillMode)];
            da.FillModeConstantVal = (double) serialized[nameof(FillModeConstantVal)];
            da.CutoutPatchPercentage = (double) serialized[nameof(CutoutPatchPercentage)];
            da.AlphaCutMix = (double) serialized[nameof(AlphaCutMix)];
            da.AlphaMixup = (double) serialized[nameof(AlphaMixup)];
            da.RotationRangeInDegrees = (double) serialized[nameof(RotationRangeInDegrees)];
            da.ZoomRange = (double)serialized[nameof(ZoomRange)];
            da.EqualizeOperationProbability = (double)serialized[nameof(EqualizeOperationProbability)];
            da.AutoContrastOperationProbability = (double)serialized[nameof(AutoContrastOperationProbability)];
            da.InvertOperationProbability = (double)serialized[nameof(InvertOperationProbability)];
            da.BrightnessOperationProbability = (double)serialized[nameof(BrightnessOperationProbability)];
            da.BrightnessOperationEnhancementFactor = (double)serialized[nameof(BrightnessOperationEnhancementFactor)];
            da.ColorOperationProbability = (double)serialized[nameof(ColorOperationProbability)];
            da.ColorOperationEnhancementFactor = (double)serialized[nameof(ColorOperationEnhancementFactor)];
            da.ContrastOperationProbability = (double)serialized[nameof(ContrastOperationProbability)];
            da.ContrastOperationEnhancementFactor = (double)serialized[nameof(ContrastOperationEnhancementFactor)];
            da.RandAugment_N = (int)serialized[nameof(RandAugment_N)];
            da.RandAugment_M = (int)serialized[nameof(RandAugment_M)];
            return da;
        }
        #endregion
    }
}
