using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class ImageDataGenerator
    {
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
            AUTO_AUGMENT_IMAGENET
        };


        #region private fields
        private readonly Random[] _rands;

        private readonly DataAugmentationEnum _dataAugmentationType;

        //randomly shift images horizontally
        private readonly double _widthShiftRangeInPercentage;
        //randomly shift images vertically
        private readonly double _heightShiftRangeInPercentage;
        //randomly flip images
        private readonly bool _horizontalFlip;
        //randomly flip images
        private readonly bool _verticalFlip;
        //set mode for filling points outside the input boundaries
        private readonly FillModeEnum _fillMode;
        //value used for fill_mode  {get;set;} = "constant"
        private readonly double _fillModeConstantVal;
        /// <summary>
        /// % of the max(width,height) of the zero mask to apply to the input picture (see: https://arxiv.org/pdf/1708.04552.pdf)
        /// recommended size : 16/32=0.5 (= 16x16) for CIFAR-10 / 8/32=0.25 (= 8x8) for CIFAR-100 / 20/32 (= 20x20) for SVHN / 32/96 (= 32x32) for STL-10
        /// less or equal to 0.0 means no cutout
        /// </summary>
        private readonly double _cutoutPatchPercentage;
        /// <summary>
        /// The alpha coefficient used for CutMix
        /// A value less or equal then 0 will disable CutMix
        /// Alpha will be used as an input of the beta law to compute lambda
        /// lambda is the % of the original to keep (1-lambda will be taken from another element and mixed with current)
        /// the % of the max(width,height) of the CutMix mask to apply to the input picture (see: https://arxiv.org/pdf/1905.04899.pdf)
        /// </summary>
        private readonly double _alphaCutMix;
        /// <summary>
        /// The alpha coefficient used for Mixup
        /// A value less or equal then 0 will disable Mixup (see: https://arxiv.org/pdf/1710.09412.pdf)
        /// </summary>
        private readonly double _alphaMixup;


        /// <summary>
        /// rotation range in degrees, in [0,180] range.
        /// The actual rotation will be a random number in [-_rotationRangeInDegrees,+_rotationRangeInDegrees]
        /// </summary>
        private readonly double _rotationRangeInDegrees;
        /// <summary>
        /// Range for random zoom. [lower, upper] = [1 - _zoomRange, 1 + _zoomRange].
        /// </summary>
        private readonly double _zoomRange;
        #endregion

        public static readonly ImageDataGenerator NoDataAugmentation = new ImageDataGenerator(DataAugmentationEnum.NO_AUGMENTATION, 0, 0, false, false, FillModeEnum.Nearest, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        public ImageDataGenerator(
            DataAugmentationEnum dataAugmentationType,
            double widthShiftRangeInPercentage, double heightShiftRangeInPercentage,
            bool horizontalFlip, bool verticalFlip, FillModeEnum fillMode, double fillModeConstantVal,
            double cutoutPatchPercentage,
            double alphaCutMix,
            double alphaMixup,
            double rotationRangeInDegrees,
            double zoomRange)
        {
            Debug.Assert(widthShiftRangeInPercentage >= 0);
            Debug.Assert(widthShiftRangeInPercentage <= 1.0);
            Debug.Assert(heightShiftRangeInPercentage >= 0);
            Debug.Assert(heightShiftRangeInPercentage <= 1.0);
            Debug.Assert(cutoutPatchPercentage <= 1.0);
            Debug.Assert(rotationRangeInDegrees >= 0);
            Debug.Assert(rotationRangeInDegrees <= 180.0);
            Debug.Assert(zoomRange >= 0);
            Debug.Assert(zoomRange <= 1.0);
            _dataAugmentationType = dataAugmentationType;
            _widthShiftRangeInPercentage = widthShiftRangeInPercentage;
            _heightShiftRangeInPercentage = heightShiftRangeInPercentage;
            _horizontalFlip = horizontalFlip;
            _verticalFlip = verticalFlip;
            _fillMode = fillMode;
            _fillModeConstantVal = fillModeConstantVal;
            _cutoutPatchPercentage = cutoutPatchPercentage;
            _alphaCutMix = alphaCutMix;
            _alphaMixup = alphaMixup;
            _rotationRangeInDegrees = rotationRangeInDegrees;
            _zoomRange = zoomRange;
            _rands = new Random[2 * Environment.ProcessorCount];
            for (int i = 0; i < _rands.Length; ++i)
            {
                _rands[i] = new Random(i);
            }
        }


        private List<Operation> GetSubPolicy(
            int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch, 
            Func<int, ImageStatistic> indexInMiniBatchToImageStatistic,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            Random rand)
        {
            switch (_dataAugmentationType)
            {
                case DataAugmentationEnum.DEFAULT:
                    return DefaultSubPolicy(indexInMiniBatch, xOriginalMiniBatch, rand);
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10:
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInMiniBatchToImageStatistic(indexInMiniBatch), rand, 0.5, 0, 0).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP:
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInMiniBatchToImageStatistic(indexInMiniBatch), rand, _cutoutPatchPercentage, _alphaCutMix, _alphaMixup).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX:
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 1.0, 0).GetSubPolicyCifar10();
                case DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP:
                    return new AutoAugment(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, indexInMiniBatchToImageStatistic(indexInMiniBatch), rand, 0, 0, 1.0).GetSubPolicyCifar10();

                case DataAugmentationEnum.AUTO_AUGMENT_IMAGENET:
                    throw new NotImplementedException("unknown DataAugmentationEnum: " + _dataAugmentationType);
                case DataAugmentationEnum.AUTO_AUGMENT_SVHN:
                    throw new NotImplementedException("unknown DataAugmentationEnum: " + _dataAugmentationType);
                case DataAugmentationEnum.NO_AUGMENTATION:
                    return new List<Operation>();
                default:
                    throw new NotImplementedException("unknown DataAugmentationEnum: "+ _dataAugmentationType); 
            }
        }

        private List<Operation> DefaultSubPolicy(int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, Random rand)
        {
            var result = new List<Operation>();

            var nbRows = xOriginalMiniBatch.Shape[2];
            var nbCols = xOriginalMiniBatch.Shape[3];
            result.Add(Rotate.ValueOf(_rotationRangeInDegrees, rand, nbRows, nbCols));

            double widthMultiplier = 1.0;
            if (_zoomRange > 0)
            {
                //random zoom multiplier in range [1.0-zoomRange, 1.0+zoomRange]
                var zoom = 2 * _zoomRange * rand.NextDouble() - _zoomRange;
                widthMultiplier = (1.0 + zoom);
                result.Add(new ShearX(widthMultiplier));
            }
            result.Add(TranslateX.ValueOf(_widthShiftRangeInPercentage, rand, nbCols));

            if (_zoomRange > 0)
            {
                double heightMultiplier = widthMultiplier;
                result.Add(new ShearY(heightMultiplier));
            }
            result.Add(TranslateY.ValueOf(_heightShiftRangeInPercentage, rand, nbRows));

            var verticalFlip = _verticalFlip && rand.Next(2) == 0;
            if (verticalFlip)
            {
                result.Add(new VerticalFlip(nbRows));
            }
            var horizontalFlip = _horizontalFlip && rand.Next(2) == 0;
            if (horizontalFlip)
            {
                result.Add(new HorizontalFlip(nbCols));
            }
            result.Add(CutMix.ValueOf(_alphaCutMix, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(Mixup.ValueOf(_alphaMixup, indexInMiniBatch, xOriginalMiniBatch, rand));
            result.Add(Cutout.ValueOf(_cutoutPatchPercentage, rand, nbRows, nbCols));
            result.RemoveAll(x => x == null);
            OperationHelper.CheckIntegrity(result);
            return result;
        }

        public void DataAugmentationForMiniBatch(
            int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch,
            CpuTensor<float> xDataAugmentedMiniBatch, 
            CpuTensor<float> yMiniBatch,
            Func<int, int> indexInMiniBatchToCategoryId,
            Func<int, ImageStatistic> indexInMiniBatchToImageStatistic,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            var rand = _rands[indexInMiniBatch % _rands.Length];
            var subPolicy = GetSubPolicy(indexInMiniBatch, xOriginalMiniBatch, indexInMiniBatchToImageStatistic, meanAndVolatilityForEachChannel, rand);
            OperationHelper.CheckIntegrity(subPolicy);
            SubPolicy.Apply(subPolicy, indexInMiniBatch, xOriginalMiniBatch, xDataAugmentedMiniBatch, yMiniBatch, indexInMiniBatchToCategoryId, meanAndVolatilityForEachChannel, _fillMode);
        }

        public bool Equals(ImageDataGenerator other, double epsilon, string id, ref string errors)
        {
            var equals = true;
            equals &= Utils.Equals(_dataAugmentationType, other._dataAugmentationType, id + ":_dataAugmentationType", ref errors);
            equals &= Utils.Equals(_widthShiftRangeInPercentage, other._widthShiftRangeInPercentage, epsilon, id + ":_widthShiftRange", ref errors);
            equals &= Utils.Equals(_heightShiftRangeInPercentage, other._heightShiftRangeInPercentage, epsilon, id + ":_heightShiftRange", ref errors);
            equals &= Utils.Equals(_horizontalFlip, other._horizontalFlip, id + ":_horizontalFlip", ref errors);
            equals &= Utils.Equals(_verticalFlip, other._verticalFlip, id + ":_verticalFlip", ref errors);
            equals &= Utils.Equals(_fillMode, other._fillMode, id + ":_fillMode", ref errors);
            equals &= Utils.Equals(_fillModeConstantVal, other._fillModeConstantVal, epsilon, id + ":_fillModeConstantVal", ref errors);
            equals &= Utils.Equals(_cutoutPatchPercentage, other._cutoutPatchPercentage, epsilon, id + ":_cutoutPatchPercentage", ref errors);
            equals &= Utils.Equals(_alphaCutMix, other._alphaCutMix, epsilon, id + ":_alphaCutMix", ref errors);
            equals &= Utils.Equals(_alphaMixup, other._alphaMixup, epsilon, id + ":_AlphaMixup", ref errors);
            equals &= Utils.Equals(_rotationRangeInDegrees, other._rotationRangeInDegrees, epsilon, id + ":_rotationRangeInDegrees", ref errors);
            equals &= Utils.Equals(_zoomRange, other._zoomRange, epsilon, id + ":_zoomRange", ref errors);
            return equals;
        }

        #region serialization
        public string Serialize()
        {
            if (!UseDataAugmentation)
            {
                return "";
            }
            return new Serializer()
                .Add(nameof(_dataAugmentationType), (int)_dataAugmentationType)
                .Add(nameof(_widthShiftRangeInPercentage), _widthShiftRangeInPercentage)
                .Add(nameof(_heightShiftRangeInPercentage), _heightShiftRangeInPercentage)
                .Add(nameof(_horizontalFlip), _horizontalFlip)
                .Add(nameof(_verticalFlip), _verticalFlip)
                .Add(nameof(_fillMode), (int)_fillMode)
                .Add(nameof(_fillModeConstantVal), _fillModeConstantVal)
                .Add(nameof(_cutoutPatchPercentage), _cutoutPatchPercentage)
                .Add(nameof(_alphaCutMix), _alphaCutMix)
                .Add(nameof(_alphaMixup), _alphaMixup)
                .Add(nameof(_rotationRangeInDegrees), _rotationRangeInDegrees)
                .Add(nameof(_zoomRange), _zoomRange)
                .ToString();
        }
        public static ImageDataGenerator ValueOf(IDictionary<string, object> serialized)
        {
            if (!serialized.ContainsKey(nameof(_widthShiftRangeInPercentage)))
            {
                return NoDataAugmentation;
            }
            return new ImageDataGenerator(
                (DataAugmentationEnum)serialized[nameof(_dataAugmentationType)],
                (double)serialized[nameof(_widthShiftRangeInPercentage)],
                (double)serialized[nameof(_heightShiftRangeInPercentage)],
                (bool)serialized[nameof(_horizontalFlip)],
                (bool)serialized[nameof(_verticalFlip)],
                (FillModeEnum)serialized[nameof(_fillMode)],
                (double)serialized[nameof(_fillModeConstantVal)],
                (double)serialized[nameof(_cutoutPatchPercentage)],
                (double)serialized[nameof(_alphaCutMix)],
                (double)serialized[nameof(_alphaMixup)], 
                (double)serialized[nameof(_rotationRangeInDegrees)], 
                (double)serialized[nameof(_zoomRange)]);
        }
        #endregion
 
        public bool UseDataAugmentation => !ReferenceEquals(this, NoDataAugmentation);
    }
}
