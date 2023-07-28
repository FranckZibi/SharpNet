using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public abstract class AbstractDataAugmentationStrategy
    {
        protected readonly int _indexInMiniBatch;
        protected readonly CpuTensor<float> _xOriginalMiniBatch;
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel;
        private readonly Lazy<ImageStatistic> _stats;
        protected readonly Random _rand;
        protected readonly double _cutoutPatchPercentage;
        protected readonly double _alphaCutMix;
        protected readonly double _alphaMixup;
        protected int NbRows => _xOriginalMiniBatch.Shape[2];
        protected int NbCols => _xOriginalMiniBatch.Shape[3];
        protected AbstractDataAugmentationStrategy(int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel, Lazy<ImageStatistic> stats, Random rand,
            double cutoutPatchPercentage, double alphaCutMix, double alphaMixup)
        {
            Debug.Assert(stats != null);
            _indexInMiniBatch = indexInMiniBatch;
            _xOriginalMiniBatch = xOriginalMiniBatch;
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
            _stats = stats;
            _rand = rand;
            _cutoutPatchPercentage = cutoutPatchPercentage;
            _alphaCutMix = alphaCutMix;
            _alphaMixup = alphaMixup;
        }

        ///  <summary>
        ///  deform a value taking into account the magnitude (=intensity) of the deformation
        ///  the initial (undeformed) value is 'undeformedValue'
        ///  </summary>
        ///  <param name="magnitudePercentage">
        ///  the intensity of the deformation in percentage in the range [0, 1]
        ///  a magnitude of 0.0 means no deformation, and will return 'undeformedValue'
        ///  a magnitude of 1.0 will return the max allowed deformation
        /// </param>
        ///  <param name="undeformedValue">the value associated with zero deformation (magnitude=0) </param>
        ///  <param name="maxDeformedValue">the max allowed deformation value (reached for magnitude=maxMagnitude) </param>
        ///  <param name="isSymmetricalComparedToUndeformedValue">
        ///  if false:
        ///     the deformed value range will be in the range [undeformedValue, maxDeformedValue]
        ///            'undeformedValue' will be reached for magnitude = 0
        ///            'maxDeformedValue' will be reached for magnitude = maxMagnitude
        ///  if true:
        ///     the deformed value range will be in the range [undeformedValue-(maxDeformedValue-undeformedValue), maxDeformedValue]
        ///            'undeformedValue' will be reached for magnitude = 0
        ///            'maxDeformedValue' will be reached for magnitude = maxMagnitude (50% chance)
        ///            'undeformedValue-(maxDeformedValue-undeformedValue)' will be reached for magnitude = maxMagnitude (50% chance)
        /// 
        ///   </param>
        ///  <returns></returns>
        private float MagnitudeToRange(float magnitudePercentage, float undeformedValue, float maxDeformedValue, bool isSymmetricalComparedToUndeformedValue)
        {
            Debug.Assert(magnitudePercentage >= 0f);
            Debug.Assert(magnitudePercentage <= 1f);
            float delta = (maxDeformedValue - undeformedValue) * magnitudePercentage;
            if (isSymmetricalComparedToUndeformedValue && ImageDataGenerator.IsEnabled(0.5, _rand))
            {
                delta = -delta;
            }

            return undeformedValue + delta;
        }

        private int MagnitudeToIntRange(float magnitudePercentage, int undeformedValue, int maxDeformedValue)
        {
            return (int)(MagnitudeToRange(magnitudePercentage, undeformedValue, maxDeformedValue, false) + 0.5);
        }

        /// <summary>
        /// Shear the image along the horizontal 'X' axis with rate magnitude
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be in range [-0.3, 0.3]
        /// a value of 0.0 means no deformation
        /// </param>
        /// <returns></returns>
        // ReSharper disable once UnusedMember.Local
        protected Operation ShearX(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var level = MagnitudeToRange(magnitudePercentage, 0.0f, 0.3f, true);
            return new ShearX(level);
        }

        /// <summary>
        /// Shear the image along the vertical 'Y' axis with rate magnitude
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be in range [-0.3, 0.3]
        /// a value of 0.0 means no deformation
        /// </param>
        /// <returns></returns>
        protected Operation ShearY(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var level = MagnitudeToRange(magnitudePercentage, 0.0f, 0.3f, true);
            return new ShearY(level);
        }

        /// <summary>
        /// Translate the image in the horizontal 'X' direction by magnitude number of pixels
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the associate deformation is in range [-150,150] (for image size 331x331)</param>
        /// <returns></returns>
        protected Operation TranslateX(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var shiftInPercentage = MagnitudeToRange(magnitudePercentage, 0, 150f / 331, true);
            var horizontalShift = (int)Math.Round(NbCols * shiftInPercentage, 0);
            return new TranslateX(horizontalShift);
        }

        /// <summary>
        /// Translate the image in the vertical 'Y' direction by magnitude number of pixels
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the associate deformation is in range [-150,150] (for image size 331x331)</param>
        /// <returns></returns>
        protected Operation TranslateY(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var shiftInPercentage = MagnitudeToRange(magnitudePercentage, 0, 150f / 331, true);
            var verticalShift = (int)Math.Round(NbRows*shiftInPercentage, 0);
            return new TranslateY(verticalShift);
        }

        /// <summary>
        /// Rotate the image magnitude degrees
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be in range [-30, 30]</param>
        /// <returns></returns>
        protected Operation Rotate(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var rotationInDegrees = MagnitudeToRange(magnitudePercentage, 0, 30, true);
            return new Rotate(rotationInDegrees, NbRows, NbCols);
        }

        /// <summary>
        /// Maximize the image contrast, by making the darkest pixel black and lightest pixel white.
        /// </summary>
        /// <param name="probability"></param>
        /// <returns></returns>
        protected Operation AutoContrast(double probability)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var threshold = _stats.Value.GetPixelThresholdByChannel(0);
            return new AutoContrast(threshold, _meanAndVolatilityForEachChannel);
        }

        /// <summary>
        /// Invert the pixels of the image
        /// </summary>
        /// <param name="probability"></param>
        /// <returns></returns>
        protected Operation Invert(double probability)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            return new Invert(_meanAndVolatilityForEachChannel);
        }

        /// <summary>
        /// Horizontal Flip of the image
        /// </summary>
        /// <param name="probability"></param>
        /// <returns></returns>
        protected Operation HorizontalFlip(double probability)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            return new HorizontalFlip(NbCols);
        }

        protected Operation VerticalFlip(double probability)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            return new VerticalFlip(NbCols);
        }
        protected Operation Rotate180Degrees(double probability)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            return new Rotate180Degrees(NbRows, NbCols);
        }

        /// <summary>
        /// Invert all pixels strictly above a threshold value of magnitude
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be  in range [255, 0]
        /// a value of 255 means no deformation</param>
        /// <returns></returns>
        protected Operation Solarize(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var threshold = MagnitudeToIntRange(magnitudePercentage, 255, 0);
            return new Solarize(threshold, _meanAndVolatilityForEachChannel);
        }

        /// <summary>
        /// Reduce the number of bits for each pixel to magnitude bits
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be in range [8, 4]
        /// a value of 8 means no deformation</param>
        /// <returns></returns>
        protected Operation Posterize(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var bitsPerPixel = MagnitudeToIntRange(magnitudePercentage, 8, 4);
            return new Posterize(bitsPerPixel, _meanAndVolatilityForEachChannel);
        }

        /// <summary>
        /// Control the contrast of the image. A magnitude=0 gives a gray image,
        /// whereas magnitude = 1 gives the original image.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be in range [0.1, 1.9]
        /// a value of 1.0 means no deformation</param>
        /// <returns></returns>
        protected Operation Contrast(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var enhancementFactor = MagnitudeToRange(magnitudePercentage, 1, 1.9f, true);
            var greyMean = _stats.Value.GreyMean(_meanAndVolatilityForEachChannel);
            return new Contrast(enhancementFactor, greyMean);
        }

        /// <summary>
        /// Adjust the color balance of the image, in a manner similar to
        /// the controls on a color TV set.A magnitude = 0 gives a black
        /// & white image, whereas magnitude=1 gives the original image.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be in range [0.1, 1.9]
        /// a value of 1.0 means no deformation</param>
        /// <returns></returns>
        protected Operation Color(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var enhancementFactor = MagnitudeToRange(magnitudePercentage, 1, 1.9f, true);
            return new Color(enhancementFactor);
        }

        /// <summary>
        /// Adjust the brightness of the image. A magnitude=0 gives a
        /// black image, whereas magnitude=1 gives the original image.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be in range [0.1, 1.9]
        /// a value of 1.0 means no deformation</param>
        /// <returns></returns>
        protected Operation Brightness(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var enhancementFactor = MagnitudeToRange(magnitudePercentage, 1, 1.9f, true);
            var blackMean = ImageDataGenerator.BlackMean(_meanAndVolatilityForEachChannel);
            return new Brightness(enhancementFactor, blackMean);
        }

        /// <summary>
        /// Adjust the sharpness of the image. A magnitude=0 gives a
        /// blurred image, whereas magnitude=1 gives the original image.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="magnitudePercentage">
        /// intensity of the deformation in percentage in range [0,1]
        /// the deformed value must be in range [0.1, 1.9]
        /// a value of 1.0 means no deformation</param>
        /// <returns></returns>
        protected Operation Sharpness(double probability, float magnitudePercentage)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }
            var enhancementFactor = MagnitudeToRange(magnitudePercentage, 1, 1.9f, true);
            //https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/ImageEnhance.html
            return new Sharpness(enhancementFactor);
        }

        /// <summary>
        /// Equalize the image histogram.
        /// </summary>
        /// <param name="probability"></param>
        /// <returns></returns>
        protected Operation Equalize(double probability)
        {
            if (!ImageDataGenerator.IsEnabled(probability, _rand))
            {
                return null;
            }

            return new Equalize(Operations.Equalize.GetOriginalPixelToEqualizedPixelByChannel(_stats.Value), _meanAndVolatilityForEachChannel);
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
        //private Operation Cutout(double probability, float magnitudePercentage, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch)
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
        //private Operation SamplePairing(double probability, float magnitudePercentage, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch)
        //{
        //    if (!IsEnabled(probability, _rand))
        //    {
        //        return null;
        //    }
        //    throw new NotImplementedException();
        //}
    }
}