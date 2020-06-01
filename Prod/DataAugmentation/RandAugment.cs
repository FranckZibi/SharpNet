using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class RandAugment : AbstractDataAugmentationStrategy
    {
        private const float MAGNITUDE_MAX = 30f;

        public RandAugment(int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel, Lazy<ImageStatistic> stats, Random rand,
            double cutoutPatchPercentage, double alphaCutMix, double alphaMixup) : 
            base(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, stats, rand, cutoutPatchPercentage, alphaCutMix, alphaMixup)
        {
        }

        /// <summary>
        /// Best parameters according to: https://arxiv.org/pdf/1909.13719.pdf
        ///     N = Number of operations
        ///     M = RandAugment Magnitude, (Max Magnitude = 30)
        ///         if M=6, then we'll use 6/30 = 20% of deformation intensity
        /// CIFAR-10:
        ///     WRN-28-2:   N=3, M=4
        ///     WRN-28-10:  N=3, M=5
        /// SVHN (small training set):
        ///     WRN-28-2:   N=3, M=9
        ///     WRN-28-10:  N=3, M=9
        /// SVHN (full training set):
        ///     WRN-28-2:   N=3, M=5
        ///     WRN-28-10:  N=3, M=7
        /// ImageNet:
        ///     ResNet50:   N=2, M=9
        /// COCO:
        ///     ResNet101:   N=1, M=5
        ///     ResNet200:   N=1, M=6
        /// </summary>
        /// <param name="N"></param>
        /// <param name="magnitude"></param>
        /// <returns></returns>
        public List<Operation> CreateSubPolicy(int N, int magnitude)
        {
            if (N <= 0)
            {
                throw new Exception("RandAugment called with N="+N+" (must be >=1)");
            }
            Debug.Assert(magnitude >= 0);
            var magnitudePercentage = magnitude / MAGNITUDE_MAX;
            var subPolicy = new List<Operation>();
            subPolicy.Add(HorizontalFlip(0.5));
            for (int i = 0; i < N; ++i)
            {
                subPolicy.Add(GetRandomOperation(magnitudePercentage));
            }
            subPolicy.Add(CutMix.ValueOf(_alphaCutMix, _indexInMiniBatch, _xOriginalMiniBatch, _rand));
            subPolicy.Add(Mixup.ValueOf(_alphaMixup, _indexInMiniBatch, _xOriginalMiniBatch, _rand));
            subPolicy.Add(Cutout.ValueOf(_cutoutPatchPercentage, _rand, NbRows, NbCols));
            subPolicy.RemoveAll(x => x == null);
            return subPolicy;
        }

        private Operation GetRandomOperation(float magnitudePercentage)
        {
            switch (_rand.Next(14))
            {
                //case 0:
                default: return null; //Identity
                case 1: return AutoContrast(1.0);
                case 2: return Equalize(1.0);
                case 3: return Rotate(1.0, magnitudePercentage);
                case 4: return Solarize(1.0, magnitudePercentage);
                case 5: return Color(1.0, magnitudePercentage);
                case 6: return Posterize(1.0, magnitudePercentage);
                case 7: return Contrast(1.0, magnitudePercentage);
                case 8: return Brightness(1.0, magnitudePercentage);
                case 9: return Sharpness(1.0, magnitudePercentage);
                case 10: return ShearX(1.0, magnitudePercentage);
                case 11: return ShearY(1.0, magnitudePercentage);
                case 12: return TranslateX(1.0, magnitudePercentage);
                case 13: return TranslateY(1.0, magnitudePercentage);
            }
        }
    }
}
