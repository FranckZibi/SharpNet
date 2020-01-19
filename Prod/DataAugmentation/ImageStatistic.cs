using System;
using System.Collections.Generic;
using SharpNet.DataAugmentation.Operations;
using SharpNet.Pictures;

namespace SharpNet.DataAugmentation
{
    public class ImageStatistic
    {
        private ImageStatistic(List<int[]> pixelCountByChannel, int[] shape)
        {
            PixelCountByChannel = pixelCountByChannel;
            Shape = shape;
        }
        ////https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/ImageOps.html
        ///// <summary>
        ///// an array of 1001 element
        ///// ThresholdForLightness[i]  is the percentage of pixels with a lightness in the range [0, 1000*i]
        ///// </summary>
        //public float[] ThresholdForLightness { get; }

        public List<int[]> PixelCountByChannel { get; }


       
        public int[] Shape { get; }

        public List<Tuple<int, int> > GetPixelThresholdByChannel(double cutoff)
        {
            var result = new List<Tuple<int, int>>();
            for (int channel = 0; channel < Shape[0]; ++channel)
            {
                result.Add(GetPixelThreshold(cutoff, channel));
            }
            return result;
        }

        private Tuple<int, int> GetPixelThreshold(double cutoff, int channel)
        {
            int count = Shape[1] * Shape[2];
            var toRemove = cutoff * count;
            var pixelCount = PixelCountByChannel[channel];
            int removed = 0;
            int darkThreshold = 0;
            for (; darkThreshold < pixelCount.Length; ++darkThreshold)
            {
                removed += pixelCount[darkThreshold];
                if (removed > toRemove)
                {
                    break;
                }
            }
            var lightThreshold = pixelCount.Length - 1;
            removed = 0;
            for (; lightThreshold >= 0; --lightThreshold)
            {
                removed += pixelCount[lightThreshold];
                if (removed > toRemove)
                {
                    break;
                }
            }
            return Tuple.Create(darkThreshold, lightThreshold);
        }

        public static ImageStatistic ValueOf(BitmapContent bmp)
        {
            var pixelCountByChannel = new List<int[]>();
            int bmpIdx = 0;
            for (int channel = 0; channel < bmp.Shape[0]; ++channel)
            {
                var count = new int[256];
                for (int indexInChannel = 0; indexInChannel < bmp.MultDim0; ++indexInChannel)
                {
                    ++count[bmp.Content[bmpIdx++]];
                }
                pixelCountByChannel.Add(count);
            }
            return new ImageStatistic(pixelCountByChannel, bmp.Shape);
        }

        public float GreyMean(List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            //we compute the average of grey
            var ponderedAverageByChannel = new List<float>();
            var pixelCount = Shape[1] * Shape[2];
            for (var channel = 0; channel < PixelCountByChannel.Count; channel++)
            {
                var count = PixelCountByChannel[channel];
                float sum = 0;
                for (int i = 0; i < count.Length; ++i)
                {
                    sum += i * count[i];
                }

                var normalizedValue = Operation.NormalizedValue(sum / pixelCount, channel, meanAndVolatilityForEachChannel);
                ponderedAverageByChannel.Add(normalizedValue);
            }
            var greyMean = Operation.GetGreyScale(ponderedAverageByChannel[0], ponderedAverageByChannel[1], ponderedAverageByChannel[2]);
            return greyMean;
        }
    }
}