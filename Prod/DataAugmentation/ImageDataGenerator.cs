using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class ImageDataGenerator
    {
        private readonly DataAugmentationSample _sample;

        //TODO: add FillModeEnum: Constant
        public enum FillModeEnum { Nearest, Reflect };
        public enum DataAugmentationEnum
        {
            DEFAULT,
            NO_AUGMENTATION,
            AUTO_AUGMENT_CIFAR10,
            AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP,
            AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX,
            AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP,
            AUTO_AUGMENT_SVHN,
            AUTO_AUGMENT_IMAGENET,
            RAND_AUGMENT,
            TIME_SERIES
        };

        public ImageDataGenerator(DataAugmentationSample sample)
        {
            _sample = sample;
        }


        private List<Operation> GetSubPolicy(int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            Func<int, Lazy<ImageStatistic>> indexInOriginalMiniBatchToImageStatistic,
            Random rand)
        {
            Debug.Assert(_sample.WidthShiftRangeInPercentage >= 0);
            Debug.Assert(_sample.WidthShiftRangeInPercentage <= 1.0);
            Debug.Assert(_sample.HeightShiftRangeInPercentage >= 0);
            Debug.Assert(_sample.HeightShiftRangeInPercentage <= 1.0);
            Debug.Assert(_sample.CutoutPatchPercentage <= 1.0);
            Debug.Assert(_sample.RotationRangeInDegrees >= 0);
            Debug.Assert(_sample.RotationRangeInDegrees <= 180.0);
            Debug.Assert(_sample.ZoomRange >= 0);
            Debug.Assert(_sample.ZoomRange <= 1.0);

            switch (_sample.DataAugmentationType)
            {
                case DataAugmentationEnum.DEFAULT:
                    return DefaultSubPolicy(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic, rand);
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10:
                    Debug.Assert(_sample.HorizontalFlip == true);
                    Debug.Assert(_sample.VerticalFlip == false);
                    Debug.Assert(_sample.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0.5, 0, 0, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP:
                    Debug.Assert(_sample.HorizontalFlip == true);
                    Debug.Assert(_sample.VerticalFlip == false);
                    Debug.Assert(_sample.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, _sample.WidthShiftRangeInPercentage, _sample.HeightShiftRangeInPercentage, _sample.CutoutPatchPercentage, _sample.AlphaCutMix, _sample.AlphaMixup, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX:
                    Debug.Assert(_sample.HorizontalFlip == true);
                    Debug.Assert(_sample.VerticalFlip == false);
                    Debug.Assert(_sample.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0, 1.0, 0, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP:
                    Debug.Assert(_sample.HorizontalFlip == true);
                    Debug.Assert(_sample.VerticalFlip == false);
                    Debug.Assert(_sample.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0, 0, 1.0, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_IMAGENET:
                    //Debug.Assert(_config.HorizontalFlip == true);
                    Debug.Assert(_sample.VerticalFlip == false);
                    //Debug.Assert(_config.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0.5, 0, 0, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicyImageNet();
                case DataAugmentationEnum.AUTO_AUGMENT_SVHN:
                    Debug.Assert(_sample.HorizontalFlip == false);
                    Debug.Assert(_sample.VerticalFlip == false);
                    //Debug.Assert(_config.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0.5, 0, 0, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicySVHN();
                case DataAugmentationEnum.RAND_AUGMENT:
                    return new RandAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0.5, 0, 0).CreateSubPolicy(_sample.RandAugment_N, _sample.RandAugment_M);
                case DataAugmentationEnum.NO_AUGMENTATION:
                    return new List<Operation>();
                default:
                    throw new NotImplementedException("unknown DataAugmentationEnum: "+ _sample.DataAugmentationType); 
            }
        }


        private List<Operation> DefaultSubPolicy(
            int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            Func<int, Lazy<ImageStatistic>> indexInMiniBatchToImageStatistic,
            Random rand)
        {
            var result = new List<Operation>();

            var lazyStats = indexInMiniBatchToImageStatistic(indexInMiniBatch);

            var nbRows = xOriginalMiniBatch.Shape[2];
            var nbCols = xOriginalMiniBatch.Shape[3];
            result.Add(Rotate.ValueOf(_sample.RotationRangeInDegrees, rand, nbRows, nbCols));

            double widthMultiplier = 1.0;
            if (_sample.ZoomRange > 0)
            {
                //random zoom multiplier in range [1.0-zoomRange, 1.0+zoomRange]
                var zoom = 2 * _sample.ZoomRange * rand.NextDouble() - _sample.ZoomRange;
                widthMultiplier = (1.0 + zoom);
                result.Add(new ShearX(widthMultiplier));
            }
            result.Add(TranslateX.ValueOf(_sample.WidthShiftRangeInPercentage, rand, nbCols));

            if (_sample.ZoomRange > 0)
            {
                double heightMultiplier = widthMultiplier;
                result.Add(new ShearY(heightMultiplier));
            }
            result.Add(TranslateY.ValueOf(_sample.HeightShiftRangeInPercentage, rand, nbRows));

            var verticalFlip = _sample.VerticalFlip && rand.Next(2) == 0;
            if (verticalFlip)
            {
                result.Add(new VerticalFlip(nbRows));
            }
            var horizontalFlip = _sample.HorizontalFlip && rand.Next(2) == 0;
            if (horizontalFlip)
            {
                result.Add(new HorizontalFlip(nbCols));
            }
            var rotate180Degrees = _sample.Rotate180Degrees && rand.Next(2) == 0;
            if (rotate180Degrees)
            {
                result.Add(new Rotate180Degrees(nbRows, nbCols));
            }
            if (IsEnabled(_sample.EqualizeOperationProbability, rand))
            {
                result.Add(new Equalize(Equalize.GetOriginalPixelToEqualizedPixelByChannel(lazyStats.Value), meanAndVolatilityForEachChannel));
            }
            if (IsEnabled(_sample.AutoContrastOperationProbability, rand))
            {
                result.Add(new AutoContrast(lazyStats.Value.GetPixelThresholdByChannel(0), meanAndVolatilityForEachChannel));
            }
            if (IsEnabled(_sample.InvertOperationProbability, rand))
            {
                result.Add(new Invert(meanAndVolatilityForEachChannel));
            }
            if (IsEnabled(_sample.BrightnessOperationProbability, rand))
            {
                result.Add(new Brightness((float)_sample.BrightnessOperationEnhancementFactor, BlackMean(meanAndVolatilityForEachChannel)));
            }
            if (IsEnabled(_sample.ColorOperationProbability, rand))
            {
                result.Add(new Color((float)_sample.ColorOperationEnhancementFactor));
            }
            if (IsEnabled(_sample.ContrastOperationProbability, rand))
            {
                result.Add(new Contrast((float)_sample.ContrastOperationEnhancementFactor, lazyStats.Value.GreyMean(meanAndVolatilityForEachChannel)));
            }
            result.Add(CutMix.ValueOf(_sample.AlphaCutMix, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(Mixup.ValueOf(_sample.AlphaMixup, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(Cutout.ValueOf(_sample.CutoutPatchPercentage, rand, nbRows, nbCols));
            result.RemoveAll(x => x == null);
#if DEBUG
            OperationHelper.CheckIntegrity(result);
#endif
            return result;
        }

        public void DataAugmentationForMiniBatch(
            int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch,
            CpuTensor<float> xDataAugmentedMiniBatch, 
            CpuTensor<float> yDataAugmentedMiniBatch,
            Func<int, int> indexInOriginalMiniBatchToCategoryIndex,
            Func<int, Lazy<ImageStatistic>> indexInOriginalMiniBatchToImageStatistic,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            Random rand,
            CpuTensor<float> xBufferForDataAugmentedMiniBatch //a temporary buffer used in the mini batch
            )
        {
            var subPolicy = GetSubPolicy(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic, rand);
#if DEBUG
            OperationHelper.CheckIntegrity(subPolicy);
#endif
            SubPolicy.Apply(subPolicy, indexInMiniBatch, xOriginalMiniBatch, xDataAugmentedMiniBatch, yDataAugmentedMiniBatch, indexInOriginalMiniBatchToCategoryIndex, _sample.FillMode, xBufferForDataAugmentedMiniBatch);
        }

        public static bool IsEnabled(double probability, Random rand)
        {
            if (probability < 1e-6)
            {
                return false;
            }
            if (probability > (1.0 - 1e-6))
            {
                return true;
            }
            return rand.NextDouble() >= probability;
        }
        public static float BlackMean(List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            var blackMean = Operation.GetGreyScale(
                Operation.NormalizedValue(0f, 0, meanAndVolatilityForEachChannel),
                Operation.NormalizedValue(0f, 1, meanAndVolatilityForEachChannel),
                Operation.NormalizedValue(0f, 2, meanAndVolatilityForEachChannel));
            return blackMean;
        }

    }
}
