using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class AutoAugment
    {
        //private readonly int _indexInMiniBatch;
        private readonly CpuTensor<float> _xOriginalMiniBatch;
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel;
        private readonly ImageStatistic _stats;
        private readonly Random _rand;
        private int NbRows => _xOriginalMiniBatch.Shape[2];
        private int NbCols => _xOriginalMiniBatch.Shape[3];

        public AutoAugment(int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel, ImageStatistic stats, Random rand)
        {
            //_indexInMiniBatch = indexInMiniBatch;
            _xOriginalMiniBatch = xOriginalMiniBatch;
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
            _stats = stats;
            _rand = rand;
        }


        public List<Operation> GetSubPolicyCifar10()
        {
            switch (_rand.Next(25))
            {
                case 0: return CreateSubPolicy(Invert(0.1), Contrast(0.2, 6));
                case 1: return CreateSubPolicy(Rotate(0.7, 2), TranslateX(0.3, 9));
                case 2: return CreateSubPolicy(Sharpness(0.8, 1), Sharpness(0.9, 3));
                case 3: return CreateSubPolicy(ShearY(0.5, 8), TranslateY(0.7, 9));
                case 4: return CreateSubPolicy(AutoContrast(0.5), Equalize(0.9));
                case 5: return CreateSubPolicy(ShearY(0.2, 7), Posterize(0.3, 7));
                case 6: return CreateSubPolicy(Color(0.4, 3), Brightness(0.6, 7));
                case 7: return CreateSubPolicy(Sharpness(0.3, 9), Brightness(0.7, 9));
                case 8: return CreateSubPolicy(Equalize(0.6), Equalize(0.5));
                case 9: return CreateSubPolicy(Contrast(0.6, 7), Sharpness(0.6, 5));
                case 10: return CreateSubPolicy(Color(0.7, 7), TranslateX(0.5, 8));
                case 11: return CreateSubPolicy(Equalize(0.3), AutoContrast(0.4));
                case 12: return CreateSubPolicy(TranslateY(0.4, 3), Sharpness(0.2, 6));
                case 13: return CreateSubPolicy(Brightness(0.9, 6), Color(0.2, 8));
                case 14: return CreateSubPolicy(Solarize(0.5, 2), Invert(0.0));
                case 15: return CreateSubPolicy(Equalize(0.2), AutoContrast(0.6));
                case 16: return CreateSubPolicy(Equalize(0.2), Equalize(0.6));
                case 17: return CreateSubPolicy(Color(0.9, 9), Equalize(0.6));
                case 18: return CreateSubPolicy(AutoContrast(0.8), Solarize(0.2, 8));
                case 19: return CreateSubPolicy(Brightness(0.1, 3), Color(0.7, 0));
                case 20: return CreateSubPolicy(Solarize(0.4, 5), AutoContrast(0.9));
                case 21: return CreateSubPolicy(TranslateY(0.9, 9), TranslateY(0.7, 9));
                case 22: return CreateSubPolicy(AutoContrast(0.9), Solarize(0.8, 3));
                case 23: return CreateSubPolicy(Equalize(0.8), Invert(0.1));
                case 24: return CreateSubPolicy(TranslateY(0.7, 9), AutoContrast(0.9));
                default: throw new Exception("Invalid random returned value");
            }
        }

        /// <summary>
        /// Shear the image along the horizontal 'X' axis with rate magnitude
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes[-0.3, 0.3]</param>
        /// <returns></returns>
        // ReSharper disable once UnusedMember.Local
        private Operation ShearX(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var widthMultiplier = MagnitudeToRange(magnitude_0_9, 0.7f, 1.3f);
            return new ShearX(widthMultiplier);
        }
        /// <summary>
        /// Shear the image along the vertical 'Y' axis with rate magnitude
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes[-0.3, 0.3]</param>
        /// <returns></returns>
        private Operation ShearY(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var heightMultiplier = MagnitudeToRange(magnitude_0_9, 0.7f, 1.3f);
            return new ShearY(heightMultiplier);
        }

        /// <summary>
        /// Translate the image in the horizontal 'X' direction by magnitude number of pixels
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [-150,150] (for image size 331x331)</param>
        /// <returns></returns>
        private Operation TranslateX(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var horizontalShiftInPercentage = MagnitudeToRange(magnitude_0_9, -150f / 331, 150f / 331);
            var horizontalShift = (int)Math.Round(NbCols * horizontalShiftInPercentage, 0);
            return new TranslateX(horizontalShift);
        }

        /// <summary>
        /// Translate the image in the vertical 'Y' direction by magnitude number of pixels
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [-150,150] (for image size 331x331)</param>
        /// <returns></returns>
        private Operation TranslateY(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var verticalShiftInPercentage = MagnitudeToRange(magnitude_0_9, -150f / 331, 150f / 331);
            var verticalShift = (int)Math.Round(NbRows*verticalShiftInPercentage, 0);
            return new TranslateY(verticalShift);
        }

        /// <summary>
        /// Rotate the image magnitude degrees
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [-30,30]</param>
        /// <returns></returns>
        private Operation Rotate(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var rotationInDegrees = MagnitudeToRange(magnitude_0_9,-30f,30f);
            return new Rotate(rotationInDegrees, NbRows, NbCols);
        }

        /// <summary>
        /// Maximize the image contrast, by making the darkest pixel black and lightest pixel white.
        /// </summary>
        /// <param name="probability"></param>
        /// <returns></returns>
        private Operation AutoContrast(double probability)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var threshold = _stats.GetPixelThresholdByChannel(0);
            return new AutoContrast(threshold, _meanAndVolatilityForEachChannel);
        }

        /// <summary>
        /// Invert the pixels of the image
        /// </summary>
        /// <param name="probability"></param>
        /// <returns></returns>
        private Operation Invert(double probability)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            return new Invert(_meanAndVolatilityForEachChannel);
        }

        /// <summary>
        /// Invert all pixels strictly above a threshold value of magnitude
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [0,255]</param>
        /// <returns></returns>
        private Operation Solarize(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var threshold = MagnitudeToRange(magnitude_0_9, 0, 255);
            return new Solarize(threshold, _meanAndVolatilityForEachChannel);
        }

        private static int MagnitudeToRange(int magnitude_0_9, int minRange, int maxRange)
        {
            Debug.Assert(maxRange>=minRange);
            return minRange + ((maxRange - minRange) * magnitude_0_9) / 9;
        }

        private static float MagnitudeToRange(int magnitude_0_9, float minRange, float maxRange)
        {
            Debug.Assert(maxRange >= minRange);
            return minRange + ((maxRange - minRange) * magnitude_0_9) / 9;
        }
        /// <summary>
        /// Reduce the number of bits for each pixel to magnitude bits
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [4,8]</param>
        /// <returns></returns>
        private Operation Posterize(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var bitsPerPixel = MagnitudeToRange(magnitude_0_9, 4, 8);
            return new Posterize(bitsPerPixel, _meanAndVolatilityForEachChannel);
        }

        /// <summary>
        /// Control the contrast of the image. A magnitude=0 gives a gray image,
        /// whereas magnitude = 1 gives the original image.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [0.1,1.9]</param>
        /// <returns></returns>
        private Operation Contrast(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var ponderedAverageByChannel =  new List<float>();
            var pixelCount = _stats.Shape[1] * _stats.Shape[2];
            for (var channel = 0; channel < _stats.PixelCountByChannel.Count; channel++)
            {
                var count = _stats.PixelCountByChannel[channel];
                float sum = 0;
                for (int i = 0; i < count.Length; ++i)
                {
                    sum += i * count[i];
                }
                var normalizedValue = Operation.NormalizedValue(sum / pixelCount, channel, _meanAndVolatilityForEachChannel);
                ponderedAverageByChannel.Add(normalizedValue);
            }
            var greyMean = Operation.GetGreyScale(ponderedAverageByChannel[0], ponderedAverageByChannel[1], ponderedAverageByChannel[2]);
            var enhancementFactor = MagnitudeToRange(magnitude_0_9, 0.1f, 1.9f);
            return new Contrast(enhancementFactor, greyMean);
        }

        /// <summary>
        /// Adjust the color balance of the image, in a manner similar to
        /// the controls on a color TV set.A magnitude = 0 gives a black
        /// & white image, whereas magnitude=1 gives the original image.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [0.1,1.9]</param>
        /// <returns></returns>
        private Operation Color(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var enhancementFactor = MagnitudeToRange(magnitude_0_9, 0.1f, 1.9f);
            return new Color(enhancementFactor);
}

        /// <summary>
        /// Adjust the brightness of the image. A magnitude=0 gives a
        /// black image, whereas magnitude=1 gives the original image.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [0.1,1.9]</param>
        /// <returns></returns>
        private Operation Brightness(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            var enhancementFactor = MagnitudeToRange(magnitude_0_9, 0.1f, 1.9f);
            var blackMean = Operation.GetGreyScale(
                Operation.NormalizedValue(0f, 0, _meanAndVolatilityForEachChannel),
                Operation.NormalizedValue(0f, 1, _meanAndVolatilityForEachChannel),
                Operation.NormalizedValue(0f, 2, _meanAndVolatilityForEachChannel));
            return new Brightness(enhancementFactor, blackMean);
        }

        /// <summary>
        /// Adjust the sharpness of the image. A magnitude=0 gives a
        /// blurred image, whereas magnitude=1 gives the original image.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitude_0_9">Range of magnitudes [0.1,1.9]</param>
        /// <returns></returns>
        private Operation Sharpness(double probability, int magnitude_0_9)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            //?D TODO
            return null;
            //https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/ImageEnhance.html
            //throw new NotImplementedException();
        }


        /// <summary>
        /// Equalize the image histogram.
        /// </summary>
        /// <param name="probability"></param>
        /// <returns></returns>
        private Operation Equalize(double probability)
        {
            if (!IsEnabled(probability, _rand))
            {
                return null;
            }
            return new Equalize(Operations.Equalize.GetOriginalPixelToEqualizedPixelByChannel(_stats), _meanAndVolatilityForEachChannel);
        }

        private static List<Operation> CreateSubPolicy(Operation op1, Operation op2)
        {
            var subPolicy = new List<Operation> { op1, op2 };
            subPolicy.RemoveAll(x => x == null);
            return subPolicy;
        }

        private static bool IsEnabled(double probability, Random rand)
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

        ///// <summary>
        ///// Set a random square patch of side-length magnitude pixels to gray.
        ///// </summary>
        ///// <param name="probability"></param>
        ///// <param name="magnitude_0_9">Range of magnitudes [0,60]</param>
        ///// <param name="indexInMiniBatch"></param>
        ///// <param name="xOriginalMiniBatch"></param>
        ///// <param name="rand"></param>
        ///// <returns></returns>
        //private Operation Cutout(double probability, int magnitude_0_9, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch)
        //{
        //    if (!IsEnabled(probability, _rand))
        //    {
        //        return null;
        //    }
        //    throw new NotImplementedException();
        //}

        ///// <summary>
        ///// Linearly add the image with another image (selected at random from the same mini-batch)
        ///// with weight magnitude, without changing the label
        ///// </summary>
        ///// <param name="probability"></param>
        ///// <param name="magnitude_0_9">Range of magnitudes[0, 0.4]</param>
        ///// <param name="indexInMiniBatch"></param>
        ///// <param name="xOriginalMiniBatch"></param>
        ///// <param name="rand"></param>
        ///// <returns></returns>
        //private Operation SamplePairing(double probability, int magnitude_0_9, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch)
        //{
        //    if (!IsEnabled(probability, _rand))
        //    {
        //        return null;
        //    }
        //    throw new NotImplementedException();
        //}
    }
}
