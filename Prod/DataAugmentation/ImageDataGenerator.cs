using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation.Operations;
using SharpNet.Networks;

namespace SharpNet.DataAugmentation
{
    public class ImageDataGenerator
    {
        private readonly NetworkSample _sample;

        //TODO: add FillModeEnum: Constant
        public enum FillModeEnum { Nearest, Reflect, Modulo };
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
            // ReSharper disable once UnusedMember.Global
            TIME_SERIES
        };

        public ImageDataGenerator(NetworkSample sample)
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
            Debug.Assert(_sample.ColumnsCutoutPatchPercentage <= 1.0);
            Debug.Assert(_sample.RowsCutoutPatchPercentage <= 1.0);
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
                    Debug.Assert(_sample.Rotate90Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0.5, 0, 0, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP:
                    Debug.Assert(_sample.HorizontalFlip == true);
                    Debug.Assert(_sample.VerticalFlip == false);
                    Debug.Assert(_sample.Rotate180Degrees == false);
                    Debug.Assert(_sample.Rotate90Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, _sample.WidthShiftRangeInPercentage, _sample.HeightShiftRangeInPercentage, _sample.CutoutPatchPercentage, _sample.AlphaCutMix, _sample.AlphaMixup, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX:
                    Debug.Assert(_sample.HorizontalFlip == true);
                    Debug.Assert(_sample.VerticalFlip == false);
                    Debug.Assert(_sample.Rotate180Degrees == false);
                    Debug.Assert(_sample.Rotate90Degrees == false);
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 0, 1.0, 0, _sample.HorizontalFlip, _sample.VerticalFlip, _sample.Rotate180Degrees).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP:
                    Debug.Assert(_sample.HorizontalFlip == true);
                    Debug.Assert(_sample.VerticalFlip == false);
                    Debug.Assert(_sample.Rotate180Degrees == false);
                    Debug.Assert(_sample.Rotate90Degrees == false);
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

            var nbRows = xOriginalMiniBatch.Shape[^2];
            var nbCols = xOriginalMiniBatch.Shape[^1];
            result.Add(Rotate.ValueOf(_sample.RotationRangeInDegrees, rand, nbRows, nbCols));

            double horizontalMultiplier = 1.0;
            if (_sample.ZoomRange > 0)
            {
                //random zoom multiplier in range [1.0-zoomRange, 1.0+zoomRange]
                var zoom = 2 * _sample.ZoomRange * rand.NextDouble() - _sample.ZoomRange;
                horizontalMultiplier = (1.0 + zoom);
                result.Add(new ShearX(horizontalMultiplier));
            }
            result.Add(TranslateX.ValueOf(_sample.WidthShiftRangeInPercentage, rand, nbCols));

            if (_sample.ZoomRange > 0)
            {
                double verticalMultiplier = horizontalMultiplier;
                result.Add(new ShearY(verticalMultiplier));
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
            var rotate90Degrees = _sample.Rotate90Degrees && rand.Next(2) == 0;
            if (rotate90Degrees)
            {
                result.Add(new Rotate90Degrees(nbRows, nbCols));
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

            //we can not use CutMix && Mixup at the same time: if they are both enabled we need to disable one of them (randomly)
            var alphaCutMix = _sample.AlphaCutMix;
            var alphaColumnsCutMix = _sample.AlphaColumnsCutMix;
            var alphaRowsCutMix = _sample.AlphaRowsCutMix;
            var alphaMixup = _sample.AlphaMixup;
            if (alphaCutMix > 0 && alphaMixup > 0)
            {
                // Mixup and CutMix can not be used at the same time: we need to disable one of them
                if (Utils.RandomCoinFlip())
                {
                    alphaMixup = 0; //We disable Mixup
                }
                else
                {
                    alphaCutMix = alphaRowsCutMix = alphaColumnsCutMix = 0; //We disable CutMix
                }
            }

            result.Add(CutMix.ValueOf(alphaCutMix, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(CutMix.ValueOfColumnsCutMix(alphaColumnsCutMix, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(CutMix.ValueOfRowsCutMix(alphaRowsCutMix, indexInMiniBatch, xOriginalMiniBatch, rand));

            result.Add(Mixup.ValueOf(alphaMixup, indexInMiniBatch, xOriginalMiniBatch, rand));
            for (int i = 0; i < _sample.CutoutCount; i++)
            {
                result.Add(Cutout.ValueOf(_sample.CutoutPatchPercentage, rand, nbRows, nbCols));
            }
            for (int i = 0; i < _sample.ColumnsCutoutCount; i++)
            {
                result.Add(Cutout.ValueOfColumnsCutout(_sample.ColumnsCutoutPatchPercentage, rand, nbRows, nbCols));
            }
            for (int i = 0; i < _sample.RowsCutoutCount; i++)
            {
                result.Add(Cutout.ValueORowsCutout(_sample.RowsCutoutPatchPercentage, rand, nbRows, nbCols));
            }

            result.RemoveAll(x => x == null);
#if DEBUG
            OperationHelper.CheckIntegrity(result);
#endif
            return result;
        }

        //special case: shape of (batchSize, rows, cols) (where there is only 1 channel that is omitted)
        public static (CpuTensor<float>, CpuTensor<float>, CpuTensor<float>) To_NCHW(CpuTensor<float> a, CpuTensor<float> b, CpuTensor<float> c)
        {
            return ((CpuTensor<float>)Tensor.To_NCHW(a), (CpuTensor<float>)Tensor.To_NCHW(b), (CpuTensor<float>)Tensor.To_NCHW(c));
        }

        public void DataAugmentationForMiniBatch(
            int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch,
            CpuTensor<float> xDataAugmentedMiniBatch,
            CpuTensor<float> yOriginalMiniBatch,
            CpuTensor<float> yDataAugmentedMiniBatch,
            Func<int, int> indexInOriginalMiniBatchToCategoryIndex,
            Func<int, Lazy<ImageStatistic>> indexInOriginalMiniBatchToImageStatistic,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            Random rand,
            CpuTensor<float> xBufferForDataAugmentedMiniBatch //a temporary buffer used in the mini batch
            )
        {
            //we ensure that all tensors shape are 4D 'NCHW' tensors  (where N is the batch size, C is the number of channels, H is the height and W is the width)
            (xOriginalMiniBatch, xDataAugmentedMiniBatch, xBufferForDataAugmentedMiniBatch) = To_NCHW(xOriginalMiniBatch, xDataAugmentedMiniBatch, xBufferForDataAugmentedMiniBatch);
            
            var subPolicy = GetSubPolicy(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInOriginalMiniBatchToImageStatistic, rand);
#if DEBUG
            OperationHelper.CheckIntegrity(subPolicy);
#endif
            SubPolicy.Apply(subPolicy, indexInMiniBatch, xOriginalMiniBatch, xDataAugmentedMiniBatch, yOriginalMiniBatch, yDataAugmentedMiniBatch, indexInOriginalMiniBatchToCategoryIndex, _sample.FillMode, xBufferForDataAugmentedMiniBatch);
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
