using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using SharpNet.CPU;

namespace SharpNet.Pictures
{
    public class BitmapContent : CpuTensor<byte>
    {
        /// <summary>
        /// Load a RGB bitmap (with 3 channels)
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="description"></param>
        /// <returns></returns>
        public static BitmapContent ValueFomSingleRgbBitmap(string filename, string description)
        {
            using (var bmp = new Bitmap(filename))
            {
                return ValueFomSingleRgbBitmap(bmp, description);
            }
        }
        /// <summary>
        /// Construct an element stacking several bitmaps, each bitmap containing a single channel
        /// </summary>
        /// <param name="singleChannelBitmaps">list of bitmap files, each containing a single channel (meaning R=G=B)</param>
        /// <param name="description"></param>
        /// <returns></returns>
        public static BitmapContent ValueFromSeveralSingleChannelBitmaps(IEnumerable<string> singleChannelBitmaps, string description)
        {
            var bmps = singleChannelBitmaps.Select(filename => new Bitmap(filename)).ToList();
            var result = ValueFromSeveralSingleChannelBitmaps(bmps, description);
            bmps.ForEach(bmp=>bmp.Dispose());
            return result;
        }
        private BitmapContent(int[] shape, byte[] data, string description) : base(shape, data, description)
        {
        }
        public int GetChannels() => Shape[0];
        public int GetHeight() => Shape[1];
        public int GetWidth() => Shape[2];
        public BitmapContent MakeSquarePictures(bool alwaysUseBiggestSideForWidthSide, Tuple<byte, byte, byte> fillingColor)
        {
            int height = Math.Max(GetHeight(), GetWidth());
            int width = height;
            var content = new byte[3 * height * width];
            var result = new BitmapContent(new[] { Shape[0], height, width }, content, Description);

            bool swapWidthAndHeight = alwaysUseBiggestSideForWidthSide && GetHeight() > GetWidth();
            int startRow = (height - GetHeight()) / 2;
            int endRow = startRow + GetHeight() - 1;
            int startCol = (width - GetWidth()) / 2;
            int endCol = startCol + GetWidth() - 1;

            for (int channel = 0; channel < Shape[0]; ++channel)
            {
                var filling = (channel == 0) ? fillingColor.Item1 : (channel == 1 ? fillingColor.Item2 : fillingColor.Item3);
                for (int row = 0; row < GetHeight(); ++row)
                {
                    for (int col = 0; col < GetWidth(); ++col)
                    {
                        result.Set(channel, row, col, filling);
                    }
                }
            }

            for (int channel = 0; channel < Shape[0]; ++channel)
            {
                for (int row = startRow; row <= endRow; ++row)
                {
                    for (int col = startCol; col <= endCol; ++col)
                    {
                        if (swapWidthAndHeight)
                        {
                            result.Set(channel, col, row, Get(channel, row - startRow, col - startCol));
                        }
                        else
                        {
                            result.Set(channel, row, col, Get(channel, row - startRow, col - startCol));
                        }
                    }
                }
            }
            return result;

        }
        public BitmapContent CropBorder()
        {
            var cropped = GetBorderCoordinates();
            return Crop(cropped.rowStart, cropped.rowEnd, cropped.colStart, cropped.colEnd);
        }
        public void Save(List<string> filenames)
        {
            if (filenames.Count == 1)
            {
                Debug.Assert(GetChannels() == 3);
                var bmp = AsBitmap();
                PictureTools.SavePng(bmp, filenames[0]);
                bmp.Dispose();
            }
            else
            {
                Debug.Assert(GetChannels() == filenames.Count);
                for (int channel = 0; channel < GetChannels(); ++channel)
                {
                    var bmp = AsBitmapForChannel(channel);
                    PictureTools.SavePng(bmp, filenames[channel]);
                    bmp.Dispose();
                }
            }
        }


      
        /// <summary>
        /// 
        /// </summary>
        /// <param name="_sum_SumSquare_Count_For_Each_Channel">
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+0] : sum of all elements in channel 'channel'
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+1] : sum of squares of all elements in channel 'channel'
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+2] : count of all elements in channel 'channel'
        /// </param>
        /// <param name="ignoreZeroPixel"></param>
        public void UpdateWith_Sum_SumSquare_Count_For_Each_Channel(float[] _sum_SumSquare_Count_For_Each_Channel, bool ignoreZeroPixel)
        {
            for (int channel = 0; channel < GetChannels(); ++channel)
            {
                var sum = 0f;
                var sumSquare = 0f;
                int count = 0;
                for (int row = 0; row < GetHeight(); ++row)
                {

                    for (int col = 0; col < GetWidth(); ++col)
                    {
                        if (ignoreZeroPixel && IzZeroPixel(row, col))
                        {
                            continue;
                        }
                        var val = Get(channel, row, col);
                        sum += val;
                        sumSquare += val * val;
                        ++count;
                    }
                }
                lock (_sum_SumSquare_Count_For_Each_Channel)
                {
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel] += sum;
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel + 1] += sumSquare;
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel + 2] += count;
                }
            }
        }
        /// <summary>
        /// returns true if the pixel (r,g,b) at index (row, col) is (0,0,0) (entirely made with zero for each channel
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public bool IzZeroPixel(int row, int col)
        {
            for (int channel = 0; channel < GetChannels(); ++channel)
            {
                if (Get(channel, row, col) != 0)
                {
                    return false;
                }
            }
            return true;
        }


        /// <summary>
        /// compute the volatility of each row
        /// </summary>
        /// <returns></returns>
        private double[] RowVolatility()
        {
            var result = new double[GetHeight()];
            for (int channel = 0; channel < GetChannels(); ++channel)
            {
                for (int row = 0; row < GetHeight(); ++row)
                {
                    double sumRow = 0.0;
                    double sumRowSquare = 0.0;
                    for (int col = 0; col < GetWidth(); ++col)
                    {
                        var element = (double)Get(channel, row, col);
                        sumRow += element;
                        sumRowSquare += element * element;
                    }

                    var mean = sumRow / GetWidth();
                    var meanSquare = mean * mean;
                    var rowVolatitily = Math.Sqrt(sumRowSquare / GetWidth() - meanSquare);
                    result[row] += rowVolatitily;
                }
            }

            return result;
        }
        /// <summary>
        /// compute the volatility of each column
        /// </summary>
        /// <returns></returns>
        private double[] ColumnVolatility()
        {
            double[] result = new double[GetWidth()];
            for (int channel = 0; channel < GetChannels(); ++channel)
            {
                for (int col = 0; col < GetWidth(); ++col)
                {
                    double sumColumn = 0.0;
                    double sumColumnSquare = 0.0;
                    for (int row = 0; row < GetHeight(); ++row)
                    {
                        var element = (double) Get(channel, row, col);
                        sumColumn += element;
                        sumColumnSquare += element * element;
                    }
                    var mean = sumColumn / GetHeight();
                    var meanSquare = mean * mean;
                    var colVolatility = Math.Sqrt(sumColumnSquare / GetHeight() - meanSquare);
                    result[col] += colVolatility;
                }
            }
            return result;
        }
        private BitmapContent Crop(int rowStart, int rowEnd, int colStart, int colEnd)
        {
            int height = rowEnd - rowStart + 1;
            int width = colEnd - colStart + 1;
            var content = new byte[3 * height * width];
            var result = new BitmapContent(new []{Shape[0], height, width}, content, Description);
            for (int channel = 0; channel < Shape[0]; ++channel)
            {
                for (int row = rowStart; row <= rowEnd; ++row)
                {
                    for (int col = colStart; col <= colEnd; ++col)
                    {
                        result.Set(channel, row-rowStart, col-colStart, Get(channel, row, col));
                    }
                }
            }
            return result;
        }
        private (int rowStart, int rowEnd, int colStart, int colEnd) GetBorderCoordinates()
        {
            (int rowStart, int rowEnd) = GetCropped(RowVolatility());
            (int colStart, int colEnd) = GetCropped(ColumnVolatility());
            return (rowStart, rowEnd, colStart, colEnd);
        }
        private static (int leftIndex, int rightIndex) GetCropped(double[] volatilities)
        {
            var length = volatilities.Length;
            var threshold = ComputeColumnVolatilityThreshold(volatilities);
            var lowerThreshold = 0.2 * threshold;


            var lowVolSegments = new List<KeyValuePair<int, int>>();



            int currentStartLowVolSegment = -1;
            for (int i = 0; i < length; ++i)
            {
                if (volatilities[i] < lowerThreshold)
                {
                    if (currentStartLowVolSegment == -1)
                    {
                        currentStartLowVolSegment = i;
                    }
                }
                else
                {
                    if (currentStartLowVolSegment != -1)
                    {
                        lowVolSegments.Add(new KeyValuePair<int, int>(currentStartLowVolSegment, i-1));
                        currentStartLowVolSegment = -1;
                    }
                }
                if (i == length - 1 && currentStartLowVolSegment != -1)
                {
                    lowVolSegments.Add(new KeyValuePair<int, int>(currentStartLowVolSegment, i));
                }
            }

            for (int segmentId = 1; segmentId < lowVolSegments.Count; ++segmentId)
            {
                int prevSegmentLength = lowVolSegments[segmentId - 1].Value - lowVolSegments[segmentId - 1].Key + 1;
                if (lowVolSegments[segmentId - 1].Key == 0)
                {
                    prevSegmentLength += length/10;
                }
                int currentSegmentLength = lowVolSegments[segmentId].Value - lowVolSegments[segmentId].Key + 1;
                if (lowVolSegments[segmentId].Value == length-1)
                {
                    currentSegmentLength += length/10;
                }
                int highVolSegmentLength = lowVolSegments[segmentId].Key - lowVolSegments[segmentId - 1].Value - 1;
                if (  (highVolSegmentLength < (Math.Min(prevSegmentLength, currentSegmentLength)/5))
                    ||(highVolSegmentLength ==1 && ((prevSegmentLength+currentSegmentLength)>=10)) )
                {
                    for (int i = lowVolSegments[segmentId - 1].Value + 1; i < lowVolSegments[segmentId].Key; ++i)
                    {
                        volatilities[i] = Math.Min(volatilities[i], lowerThreshold);
                    }
                    lowVolSegments[segmentId - 1] = new KeyValuePair<int, int>(lowVolSegments[segmentId - 1].Key, lowVolSegments[segmentId].Value);
                    lowVolSegments.RemoveAt(segmentId);
                    segmentId = 0;
                }
            }



            int leftIndex = 0;
            var sumFromleft = 0.0;
            for (; leftIndex < length - 1; ++leftIndex)
            {
                var vol = volatilities[leftIndex];
                var averageFromStart = sumFromleft / (1+leftIndex);
                if ( (vol > threshold) ||(averageFromStart>0 && vol > (3.0 * averageFromStart) && vol > lowerThreshold) )
                {
                    break;
                }
                sumFromleft += vol;
            }
            int rightIndex = length - 1;
            var sumFromRight = 0.0;
            for (; rightIndex > leftIndex; --rightIndex)
            {
                var vol = volatilities[rightIndex];
                var averageFromStart = sumFromRight / (length - rightIndex);
                if ((vol > threshold) || (averageFromStart > 0 && vol > (3.0 * averageFromStart) && vol > lowerThreshold))
                {
                    break;
                }
                sumFromRight += vol;
            }
            return (Math.Max(0, leftIndex), Math.Min(rightIndex, length - 1));
        }
        /// <summary>
        /// For each column, will return the volatility threshold
        /// A column with a volatility less then this threshold will be considered as constant
        /// </summary>
        /// <param name="volatilities"></param>
        /// <returns></returns>
        private static double ComputeColumnVolatilityThreshold(double[] volatilities)
        {
            var sorted = new List<double>(volatilities.Where(x=>x>=0.5));
            if (sorted.Count == 0)
            {
                sorted = new List<double>(volatilities);
            }
            sorted.Sort();
            var topTenPercentVolatility = sorted[(9 * sorted.Count) / 10];
            return topTenPercentVolatility / 10;
        }
        private Bitmap AsBitmap()
        {
            Debug.Assert(GetChannels() == 3);
            var bmp = new Bitmap(GetWidth(), GetHeight(), PixelFormat.Format24bppRgb);
            var rect = new Rectangle(0, 0, GetWidth(), GetHeight());
            // Lock the bitmap bits.  
            var bmpData = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            var rgbValues = new byte[bmpData.Stride * bmpData.Height];
            int index = 0;
            for (int row = 0; row < bmpData.Height; row++)
            {
                for (int col = 0; col < bmpData.Width; col++)
                {
                    rgbValues[index + 2] = Get(0, row, col);    //R
                    rgbValues[index + 1] = Get(1, row, col);    //G
                    rgbValues[index] = Get(2, row, col);        //B
                    index += 3;
                }
                index += bmpData.Stride - bmpData.Width * 3;
            }
            Marshal.Copy(rgbValues, 0, bmpData.Scan0, rgbValues.Length);
            // Unlock the bits.
            bmp.UnlockBits(bmpData);
            return bmp;
        }
        private Bitmap AsBitmapForChannel(int channel)
        {
            Debug.Assert(GetChannels() == 3);
            var bmp = new Bitmap(GetWidth(), GetHeight(), PixelFormat.Format24bppRgb);
            var rect = new Rectangle(0, 0, GetWidth(), GetHeight());
            // Lock the bitmap bits.  
            var bmpData = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            var rgbValues = new byte[bmpData.Stride * bmpData.Height];
            int index = 0;
            for (int row = 0; row < bmpData.Height; row++)
            {
                for (int col = 0; col < bmpData.Width; col++)
                {
                    var rgbValue = Get(channel, row, col);
                    rgbValues[index + 2] = rgbValue;    //R
                    rgbValues[index + 1] = rgbValue;    //G
                    rgbValues[index] = rgbValue;        //B
                    index += 3;
                }
                index += bmpData.Stride - bmpData.Width * 3;
            }
            Marshal.Copy(rgbValues, 0, bmpData.Scan0, rgbValues.Length);
            // Unlock the bits.
            bmp.UnlockBits(bmpData);
            return bmp;
        }

        /// <summary>
        /// Construct a Volume from several bitmaps: each contain a single (grey scale) channel
        /// </summary>
        /// <param name="bmps">List of bitmap, one per channel</param>
        /// <param name="description"></param>
        /// <returns></returns>
        private static BitmapContent ValueFromSeveralSingleChannelBitmaps(IReadOnlyList<Bitmap> bmps, string description)
        {
            var width = bmps[0].Width;
            var height = bmps[0].Height;
            var shape = new[] { bmps.Count, height, width };
            var result = new BitmapContent(shape, null, description);

            for (var channel = 0; channel < bmps.Count; channel++)
            {
                var bmpForChannel = bmps[channel];
                Debug.Assert(bmpForChannel.Height == height);
                Debug.Assert(bmpForChannel.Width == width);
                var rect = new Rectangle(0, 0, width, height);
                var bmpData = bmpForChannel.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                var stride = bmpData.Stride;
                unsafe
                {
                    var imgPtr = (byte*) (bmpData.Scan0);
                    for (int row = 0; row < height; row++)
                    {
                        for (int col = 0; col < width; col++)
                        {
                            result.Set(channel, row, col, *(imgPtr + 2)); //R
                            //We ensure that it is a grey scale bitmap (Red = Green = Blue)
                            Debug.Assert(*(imgPtr + 2) == *(imgPtr + 1));   //Red byte == Green Byte
                            Debug.Assert(*(imgPtr + 2) == *(imgPtr));       //Red byte == Blue Byte
                            imgPtr += 3;
                        }
                        imgPtr += stride - width * 3;
                    }
                }
                // Unlock the bits.
                bmpForChannel.UnlockBits(bmpData);
            }
            return result;
        }

        private static BitmapContent ValueFomSingleRgbBitmap(Bitmap bmp, string description)
        {
            var width = bmp.Width;
            var height = bmp.Height;
            var rect = new Rectangle(0, 0, width, height);
            var bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            var shape = new []{3, height, width};
            var result = new BitmapContent(shape, null, description);
            var stride = bmpData.Stride;

            unsafe
            {
                var imgPtr = (byte*)(bmpData.Scan0);

                for (int row = 0; row < height; row++)
                {
                    for (int col = 0; col < width; col++)
                    {
                        result.Set(0,row, col, *(imgPtr + 2)); //R
                        result.Set(1,row, col, *(imgPtr + 1)); //G
                        result.Set(2,row, col, *(imgPtr + 0)); //B
                        imgPtr += 3;
                    }
                    imgPtr += stride - width * 3;
                }
            }
            // Unlock the bits.
            bmp.UnlockBits(bmpData);
            return result;
        }
    }
}
