using System;
using System.Collections.Generic;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class AutoAugment : AbstractDataAugmentationStrategy
    {
        private const float MAGNITUDE_MAX = 9f;
        private readonly double _widthShiftRangeInPercentage;
        private readonly double _heightShiftRangeInPercentage;
        private readonly bool _horizontalFlip;
        private readonly bool _verticalFlip;
        private readonly bool _rotate180Degrees;

        public AutoAugment(int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel, Lazy<ImageStatistic> stats, Random rand,
            double widthShiftRangeInPercentage, double heightShiftRangeInPercentage, double cutoutPatchPercentage,
            double alphaCutMix, double alphaMixup, bool horizontalFlip, bool verticalFlip, bool rotate180Degrees) : 
            base(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, stats, rand, cutoutPatchPercentage, alphaCutMix, alphaMixup)
        {
            _widthShiftRangeInPercentage = widthShiftRangeInPercentage;
            _heightShiftRangeInPercentage = heightShiftRangeInPercentage;
            _horizontalFlip = horizontalFlip;
            _verticalFlip = verticalFlip;
            _rotate180Degrees = rotate180Degrees;
        }

        public List<Operation> GetSubPolicyCifar10()
        {
            switch (_rand.Next(25))
            {
                case 0: return CreateSubPolicy(Invert(0.1), Contrast(0.2, 6 / MAGNITUDE_MAX));
                case 1: return CreateSubPolicy(Rotate(0.7, 2 / MAGNITUDE_MAX), TranslateX(0.3, 9 / MAGNITUDE_MAX));
                case 2: return CreateSubPolicy(Sharpness(0.8, 1 / MAGNITUDE_MAX), Sharpness(0.9, 3 / MAGNITUDE_MAX));
                case 3: return CreateSubPolicy(ShearY(0.5, 8 / MAGNITUDE_MAX), TranslateY(0.7, 9 / MAGNITUDE_MAX));
                case 4: return CreateSubPolicy(AutoContrast(0.5), Equalize(0.9));
                case 5: return CreateSubPolicy(ShearY(0.2, 7 / MAGNITUDE_MAX), Posterize(0.3, 7 / MAGNITUDE_MAX));
                case 6: return CreateSubPolicy(Color(0.4, 3 / MAGNITUDE_MAX), Brightness(0.6, 7 / MAGNITUDE_MAX));
                case 7: return CreateSubPolicy(Sharpness(0.3, 9 / MAGNITUDE_MAX), Brightness(0.7, 9 / MAGNITUDE_MAX));
                case 8: return CreateSubPolicy(Equalize(0.6), Equalize(0.5));
                case 9: return CreateSubPolicy(Contrast(0.6, 7 / MAGNITUDE_MAX), Sharpness(0.6, 5 / MAGNITUDE_MAX));
                case 10: return CreateSubPolicy(Color(0.7, 7 / MAGNITUDE_MAX), TranslateX(0.5, 8 / MAGNITUDE_MAX));
                case 11: return CreateSubPolicy(Equalize(0.3), AutoContrast(0.4));
                case 12: return CreateSubPolicy(TranslateY(0.4, 3 / MAGNITUDE_MAX), Sharpness(0.2, 6 / MAGNITUDE_MAX));
                case 13: return CreateSubPolicy(Brightness(0.9, 6 / MAGNITUDE_MAX), Color(0.2, 8 / MAGNITUDE_MAX));
                case 14: return CreateSubPolicy(Solarize(0.5, 2 / MAGNITUDE_MAX), Invert(0.0));
                case 15: return CreateSubPolicy(Equalize(0.2), AutoContrast(0.6));
                case 16: return CreateSubPolicy(Equalize(0.2), Equalize(0.6));
                case 17: return CreateSubPolicy(Color(0.9, 9 / MAGNITUDE_MAX), Equalize(0.6));
                case 18: return CreateSubPolicy(AutoContrast(0.8), Solarize(0.2, 8 / MAGNITUDE_MAX));
                case 19: return CreateSubPolicy(Brightness(0.1, 3 / MAGNITUDE_MAX), Color(0.7, 0 / MAGNITUDE_MAX));
                case 20: return CreateSubPolicy(Solarize(0.4, 5 / MAGNITUDE_MAX), AutoContrast(0.9));
                case 21: return CreateSubPolicy(TranslateY(0.9, 9 / MAGNITUDE_MAX), TranslateY(0.7, 9 / MAGNITUDE_MAX));
                case 22: return CreateSubPolicy(AutoContrast(0.9), Solarize(0.8, 3 / MAGNITUDE_MAX));
                case 23: return CreateSubPolicy(Equalize(0.8), Invert(0.1));
                case 24: return CreateSubPolicy(TranslateY(0.7, 9 / MAGNITUDE_MAX), AutoContrast(0.9));
                default: throw new Exception("Invalid random returned value");
            }
        }

        public List<Operation> GetSubPolicySVHN()
        {
            switch (_rand.Next(25))
            {
                case 0: return CreateSubPolicy(ShearX(0.9, 4 / MAGNITUDE_MAX), Invert(0.2));
                case 1: return CreateSubPolicy(ShearY(0.9, 8 / MAGNITUDE_MAX), Invert(0.7));
                case 2: return CreateSubPolicy(Invert(0.6), Solarize(0.6, 6 / MAGNITUDE_MAX));
                case 3: return CreateSubPolicy(Invert(0.9), Invert(0.6));
                case 4: return CreateSubPolicy(Invert(0.6), Rotate(0.9, 3 / MAGNITUDE_MAX));
                case 5: return CreateSubPolicy(ShearX(0.9, 4 / MAGNITUDE_MAX), AutoContrast(0.8));
                case 6: return CreateSubPolicy(ShearY(0.9, 8 / MAGNITUDE_MAX), Invert(0.4));
                case 7: return CreateSubPolicy(ShearY(0.9, 5 / MAGNITUDE_MAX), Solarize(0.2, 6 / MAGNITUDE_MAX));
                case 8: return CreateSubPolicy(Invert(0.9), AutoContrast(0.8));
                case 9: return CreateSubPolicy(Invert(0.6), Rotate(0.9, 3 / MAGNITUDE_MAX));
                case 10: return CreateSubPolicy(ShearX(0.9, 4 / MAGNITUDE_MAX), Solarize(0.3, 3 / MAGNITUDE_MAX));
                case 11: return CreateSubPolicy(ShearY(0.8, 8 / MAGNITUDE_MAX), Invert(0.7));
                case 12: return CreateSubPolicy(Invert(0.9), TranslateY(0.6, 6 / MAGNITUDE_MAX));
                case 13: return CreateSubPolicy(Invert(0.9), Invert(0.6));
                case 14: return CreateSubPolicy(Contrast(0.3, 3 / MAGNITUDE_MAX), Rotate(0.8, 4 / MAGNITUDE_MAX));
                case 15: return CreateSubPolicy(Invert(0.8), TranslateY(0.0, 2 / MAGNITUDE_MAX));
                case 16: return CreateSubPolicy(ShearY(0.7, 6 / MAGNITUDE_MAX), Solarize(0.4, 8 / MAGNITUDE_MAX));
                case 17: return CreateSubPolicy(Invert(0.6), Rotate(0.8, 4 / MAGNITUDE_MAX));
                case 18: return CreateSubPolicy(ShearY(0.3, 7 / MAGNITUDE_MAX), TranslateX(0.9, 3 / MAGNITUDE_MAX));
                case 19: return CreateSubPolicy(ShearX(0.1, 6 / MAGNITUDE_MAX), Invert(0.6));
                case 20: return CreateSubPolicy(Solarize(0.7, 2 / MAGNITUDE_MAX), TranslateY(0.6, 7 / MAGNITUDE_MAX));
                case 21: return CreateSubPolicy(ShearY(0.8, 4 / MAGNITUDE_MAX), Invert(0.8));
                case 22: return CreateSubPolicy(ShearX(0.7, 9 / MAGNITUDE_MAX), TranslateY(0.8, 3 / MAGNITUDE_MAX));
                case 23: return CreateSubPolicy(ShearY(0.8, 5 / MAGNITUDE_MAX), AutoContrast(0.7));
                case 24: return CreateSubPolicy(ShearX(0.7, 2 / MAGNITUDE_MAX), Invert(0.1));
                default: throw new Exception("Invalid random returned value");
            }
        }

        public List<Operation> GetSubPolicyImageNet()
        {
            switch (_rand.Next(25))
            {
                case 0: return CreateSubPolicy(Posterize(0.4, 8 / MAGNITUDE_MAX), Rotate(0.6, 9 / MAGNITUDE_MAX));
                case 1: return CreateSubPolicy(Solarize(0.6, 5 / MAGNITUDE_MAX), AutoContrast(0.6));
                case 2: return CreateSubPolicy(Equalize(0.8), Equalize(0.6));
                case 3: return CreateSubPolicy(Posterize(0.6, 7 / MAGNITUDE_MAX), Posterize(0.6, 6 / MAGNITUDE_MAX));
                case 4: return CreateSubPolicy(Equalize(0.4), Solarize(0.2, 4 / MAGNITUDE_MAX));
                case 5: return CreateSubPolicy(Equalize(0.4), Rotate(0.8, 8 / MAGNITUDE_MAX));
                case 6: return CreateSubPolicy(Solarize(0.6, 3 / MAGNITUDE_MAX), Equalize(0.6));
                case 7: return CreateSubPolicy(Posterize(0.8, 5 / MAGNITUDE_MAX), Equalize(1.0));
                case 8: return CreateSubPolicy(Rotate(0.2, 3 / MAGNITUDE_MAX), Solarize(0.6, 8 / MAGNITUDE_MAX));
                case 9: return CreateSubPolicy(Equalize(0.6), Posterize(0.4, 6 / MAGNITUDE_MAX));
                case 10: return CreateSubPolicy(Rotate(0.8, 8 / MAGNITUDE_MAX), Color(0.4, 0 / MAGNITUDE_MAX));
                case 11: return CreateSubPolicy(Rotate(0.4, 9 / MAGNITUDE_MAX), Equalize(0.6));
                case 12: return CreateSubPolicy(Equalize(0.0), Equalize(0.8));
                case 13: return CreateSubPolicy(Invert(0.6), Equalize(1.0));
                case 14: return CreateSubPolicy(Color(0.6, 4 / MAGNITUDE_MAX), Contrast(1.0, 8 / MAGNITUDE_MAX));
                case 15: return CreateSubPolicy(Rotate(0.8, 8 / MAGNITUDE_MAX), Color(1.0, 2 / MAGNITUDE_MAX));
                case 16: return CreateSubPolicy(Color(0.8, 8 / MAGNITUDE_MAX), Solarize(0.8, 7 / MAGNITUDE_MAX));
                case 17: return CreateSubPolicy(Sharpness(0.4, 7 / MAGNITUDE_MAX), Invert(0.6));
                case 18: return CreateSubPolicy(ShearX(0.6, 5 / MAGNITUDE_MAX), Equalize(1.0));
                case 19: return CreateSubPolicy(Color(0.4, 0 / MAGNITUDE_MAX), Equalize(0.6));
                case 20: return CreateSubPolicy(Equalize(0.4), Solarize(0.2, 4 / MAGNITUDE_MAX));
                case 21: return CreateSubPolicy(Solarize(0.6, 5 / MAGNITUDE_MAX), AutoContrast(0.6));
                case 22: return CreateSubPolicy(Invert(0.6), Equalize(1.0));
                case 23: return CreateSubPolicy(Color(0.6, 4 / MAGNITUDE_MAX), Contrast(1.0, 8 / MAGNITUDE_MAX));
                case 24: return CreateSubPolicy(Equalize(0.8), Equalize(0.6));
                default: throw new Exception("Invalid random returned value");
            }
        }

        private List<Operation> CreateSubPolicy(Operation op1, Operation op2)
        {
            var subPolicy = new List<Operation>
                            {
                                Operations.TranslateX.ValueOf(_widthShiftRangeInPercentage, _rand, NbCols),
                                Operations.TranslateY.ValueOf(_heightShiftRangeInPercentage, _rand, NbRows),
                                HorizontalFlip(_horizontalFlip?0.5:0.0),
                                VerticalFlip(_verticalFlip?0.5:0.0),
                                Rotate180Degrees(_rotate180Degrees?0.5:0.0),
                                op1,
                                op2,
                                CutMix.ValueOf(_alphaCutMix, _indexInMiniBatch, _xOriginalMiniBatch, _rand),
                                Mixup.ValueOf(_alphaMixup, _indexInMiniBatch, _xOriginalMiniBatch, _rand),
                                Cutout.ValueOf(_cutoutPatchPercentage, _rand, NbRows, NbCols)
                            };
            subPolicy.RemoveAll(x => x == null);
            #if DEBUG
            OperationHelper.CheckIntegrity(subPolicy);
            #endif
            return subPolicy;
        }
    }
}
