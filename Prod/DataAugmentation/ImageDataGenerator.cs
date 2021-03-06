﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class ImageDataGenerator
    {
        private readonly DataAugmentationConfig _config;

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
            RAND_AUGMENT
        };

        public ImageDataGenerator(DataAugmentationConfig config)
        {
            _config = config;
        }


        private List<Operation> GetSubPolicy(int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            Func<int, Lazy<ImageStatistic>> indexInOriginalMiniBatchToImageStatistic,
            Random rand)
        {
            Debug.Assert(_config.WidthShiftRangeInPercentage >= 0);
            Debug.Assert(_config.WidthShiftRangeInPercentage <= 1.0);
            Debug.Assert(_config.HeightShiftRangeInPercentage >= 0);
            Debug.Assert(_config.HeightShiftRangeInPercentage <= 1.0);
            Debug.Assert(_config.CutoutPatchPercentage <= 1.0);
            Debug.Assert(_config.RotationRangeInDegrees >= 0);
            Debug.Assert(_config.RotationRangeInDegrees <= 180.0);
            Debug.Assert(_config.ZoomRange >= 0);
            Debug.Assert(_config.ZoomRange <= 1.0);

            switch (_config.DataAugmentationType)
            {
                case DataAugmentationEnum.DEFAULT:
                    return DefaultSubPolicy(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic, rand);
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10:
                    Debug.Assert(_config.HorizontalFlip == true);
                    Debug.Assert(_config.VerticalFlip == false);
                    Debug.Assert(_config.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0.5, 0, 0, _config.HorizontalFlip, _config.VerticalFlip, _config.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP:
                    Debug.Assert(_config.HorizontalFlip == true);
                    Debug.Assert(_config.VerticalFlip == false);
                    Debug.Assert(_config.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, _config.WidthShiftRangeInPercentage, _config.HeightShiftRangeInPercentage, _config.CutoutPatchPercentage, _config.AlphaCutMix, _config.AlphaMixup, _config.HorizontalFlip, _config.VerticalFlip, _config.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX:
                    Debug.Assert(_config.HorizontalFlip == true);
                    Debug.Assert(_config.VerticalFlip == false);
                    Debug.Assert(_config.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0, 1.0, 0, _config.HorizontalFlip, _config.VerticalFlip, _config.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP:
                    Debug.Assert(_config.HorizontalFlip == true);
                    Debug.Assert(_config.VerticalFlip == false);
                    Debug.Assert(_config.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0, 0, 1.0, _config.HorizontalFlip, _config.VerticalFlip, _config.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_IMAGENET:
                    //Debug.Assert(_config.HorizontalFlip == true);
                    Debug.Assert(_config.VerticalFlip == false);
                    //Debug.Assert(_config.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0.5, 0, 0, _config.HorizontalFlip, _config.VerticalFlip, _config.Rotate180Degrees).GetSubPolicyImageNet();
                case DataAugmentationEnum.AUTO_AUGMENT_SVHN:
                    Debug.Assert(_config.HorizontalFlip == false);
                    Debug.Assert(_config.VerticalFlip == false);
                    //Debug.Assert(_config.Rotate180Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0.5, 0, 0, _config.HorizontalFlip, _config.VerticalFlip, _config.Rotate180Degrees).GetSubPolicySVHN();
                case DataAugmentationEnum.RAND_AUGMENT:
                    return new RandAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0.5, 0, 0).CreateSubPolicy(_config.RandAugment_N, _config.RandAugment_M);
                case DataAugmentationEnum.NO_AUGMENTATION:
                    return new List<Operation>();
                default:
                    throw new NotImplementedException("unknown DataAugmentationEnum: "+ _config.DataAugmentationType); 
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
            result.Add(Rotate.ValueOf(_config.RotationRangeInDegrees, rand, nbRows, nbCols));

            double widthMultiplier = 1.0;
            if (_config.ZoomRange > 0)
            {
                //random zoom multiplier in range [1.0-zoomRange, 1.0+zoomRange]
                var zoom = 2 * _config.ZoomRange * rand.NextDouble() - _config.ZoomRange;
                widthMultiplier = (1.0 + zoom);
                result.Add(new ShearX(widthMultiplier));
            }
            result.Add(TranslateX.ValueOf(_config.WidthShiftRangeInPercentage, rand, nbCols));

            if (_config.ZoomRange > 0)
            {
                double heightMultiplier = widthMultiplier;
                result.Add(new ShearY(heightMultiplier));
            }
            result.Add(TranslateY.ValueOf(_config.HeightShiftRangeInPercentage, rand, nbRows));

            var verticalFlip = _config.VerticalFlip && rand.Next(2) == 0;
            if (verticalFlip)
            {
                result.Add(new VerticalFlip(nbRows));
            }
            var horizontalFlip = _config.HorizontalFlip && rand.Next(2) == 0;
            if (horizontalFlip)
            {
                result.Add(new HorizontalFlip(nbCols));
            }
            var rotate180Degrees = _config.Rotate180Degrees && rand.Next(2) == 0;
            if (rotate180Degrees)
            {
                result.Add(new Rotate180Degrees(nbRows, nbCols));
            }
            if (IsEnabled(_config.EqualizeOperationProbability, rand))
            {
                result.Add(new Equalize(Equalize.GetOriginalPixelToEqualizedPixelByChannel(lazyStats.Value), meanAndVolatilityForEachChannel));
            }
            if (IsEnabled(_config.AutoContrastOperationProbability, rand))
            {
                result.Add(new AutoContrast(lazyStats.Value.GetPixelThresholdByChannel(0), meanAndVolatilityForEachChannel));
            }
            if (IsEnabled(_config.InvertOperationProbability, rand))
            {
                result.Add(new Invert(meanAndVolatilityForEachChannel));
            }
            if (IsEnabled(_config.BrightnessOperationProbability, rand))
            {
                result.Add(new Brightness((float)_config.BrightnessOperationEnhancementFactor, BlackMean(meanAndVolatilityForEachChannel)));
            }
            if (IsEnabled(_config.ColorOperationProbability, rand))
            {
                result.Add(new Color((float)_config.ColorOperationEnhancementFactor));
            }
            if (IsEnabled(_config.ContrastOperationProbability, rand))
            {
                result.Add(new Contrast((float)_config.ContrastOperationEnhancementFactor, lazyStats.Value.GreyMean(meanAndVolatilityForEachChannel)));
            }
            result.Add(CutMix.ValueOf(_config.AlphaCutMix, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(Mixup.ValueOf(_config.AlphaMixup, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(Cutout.ValueOf(_config.CutoutPatchPercentage, rand, nbRows, nbCols));
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
            SubPolicy.Apply(subPolicy, indexInMiniBatch, xOriginalMiniBatch, xDataAugmentedMiniBatch, yDataAugmentedMiniBatch, indexInOriginalMiniBatchToCategoryIndex, _config.FillMode, xBufferForDataAugmentedMiniBatch);
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
