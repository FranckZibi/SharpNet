using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using PhotoSauce.MagicScaler;
using SharpNet.CPU;
using System;
using System.Runtime.InteropServices;

namespace SharpNet.Pictures
{


    public class BitmapContent : CpuTensor<byte>
    {
        /// <summary>
        /// Load a RGB bitmap (with 3 channels)
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static BitmapContent ValueFomSingleRgbBitmap(string filename)
        {
            using var bmp = new Bitmap(filename);
            return ValueFomSingleRgbBitmap(bmp);
        }
        /// <summary>
        /// Construct an element stacking several bitmaps, each bitmap containing a single channel
        /// </summary>
        /// <param name="singleChannelBitmaps">list of bitmap files, each containing a single channel (meaning R=G=B)</param>
        /// <returns></returns>
        public static BitmapContent ValueFromSeveralSingleChannelBitmaps(IEnumerable<string> singleChannelBitmaps)
        {
            var bmps = singleChannelBitmaps.Select(filename => new Bitmap(filename)).ToList();
            var result = ValueFromSeveralSingleChannelBitmaps(bmps);
            bmps.ForEach(bmp=>bmp.Dispose());
            return result;
        }
        public BitmapContent(int[] shape, byte[] data) : base(shape, data)
        {
        }
        public int GetChannels() => Shape[0];
        public int GetHeight() => Shape[1];
        public int GetWidth() => Shape[2];
        public BitmapContent MakeSquarePictures(bool alwaysUseBiggestSideForWidthSide, bool alwaysCropInsidePicture, Tuple<byte, byte, byte> fillingColor)
        {
            int heightAndWidth = alwaysCropInsidePicture
                ? Math.Min(GetHeight(), GetWidth())
                : Math.Max(GetHeight(), GetWidth());
            var content = new byte[3 * heightAndWidth * heightAndWidth];
            var result = new BitmapContent(new[] { Shape[0], heightAndWidth, heightAndWidth }, content);
            bool swapWidthAndHeight = alwaysUseBiggestSideForWidthSide && GetHeight() > GetWidth();
            
            for (int channel = 0; channel < Shape[0]; ++channel)
            {
                var filling = (channel == 0) ? fillingColor.Item1 : (channel == 1 ? fillingColor.Item2 : fillingColor.Item3);
                for (int row = 0; row < heightAndWidth; ++row)
                {
                    int originalRow = row + (GetHeight()-heightAndWidth) / 2;
                    for (int col = 0; col < heightAndWidth; ++col)
                    {
                        int originalCol = col + (GetWidth() - heightAndWidth) / 2;
                        var originalVal = (originalRow >= 0 && originalCol >= 0 && originalRow < GetHeight() && originalCol < GetWidth())
                            ? Get(channel, originalRow, originalCol)
                            : filling;
                        if (swapWidthAndHeight)
                        {
                            result.Set(channel, col, row, originalVal);
                        }
                        else
                        {
                            result.Set(channel, row, col, originalVal);
                        }
                    }
                }
            }
            return result;

        }
        /// <summary>
        /// return the SHA-1 of a bitmap (160 bits stored in a string of 40 bytes in hexadecimal format: 0=>f)
        /// </summary>
        public string SHA1()
        {
            var byteArray = Content.ToArray();
            using var sha1 = new SHA1CryptoServiceProvider();
            var hashBytes = sha1.ComputeHash(byteArray);
            return BitConverter.ToString(hashBytes).Replace("-", "");
        }
        public void Save(List<string> fileNames)
        {
            if (fileNames.Count == 1)
            {
                Debug.Assert(GetChannels() == 3);
                var bmp = AsBitmap();
                PictureTools.SavePng(bmp, fileNames[0]);
                bmp.Dispose();
            }
            else
            {
                Debug.Assert(GetChannels() == fileNames.Count);
                for (int channel = 0; channel < GetChannels(); ++channel)
                {
                    var bmp = AsBitmapForChannel(channel);
                    PictureTools.SavePng(bmp, fileNames[channel]);
                    bmp.Dispose();
                }
            }
        }
        public RGBColor AverageColor(RGBColorFactoryWithCache cache)
        {
            var acc = new ColorAccumulator();
            Debug.Assert(3 == Shape[0]);
            var content = SpanContent;
            for (int i = 0; i < MultDim0; ++i)
            {
                acc.Add(cache.Build(content[i], content[i+MultDim0], content[i + 2*MultDim0]));
            }
            return acc.Average;
        }
        /// <summary>
        /// Compute Sum / Sum^2 / Count of each channel.
        /// THis will be used to compute Mean/Volatility of each channel
        /// </summary>
        /// <param name="_sum_SumSquare_Count_For_Each_Channel">
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+0] : sum of all elements in channel 'channel'
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+1] : sum of squares of all elements in channel 'channel'
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+2] : count of all elements in channel 'channel'
        /// </param>
        public void UpdateWith_Sum_SumSquare_Count_For_Each_Channel(float[] _sum_SumSquare_Count_For_Each_Channel)
        {
            var content = ReadonlyContent;
            for (int channel = 0; channel < GetChannels(); ++channel)
            {
                var sum = 0f;
                var sumSquare = 0f;
                var startIdx = Idx(channel, 0, 0);
                var endIdxExcluded = Idx(channel+1, 0, 0);
                for (int idx = startIdx; idx< endIdxExcluded; ++idx)
                {
                    var val = content[idx];
                    sum += val;
                    sumSquare += val * val;
                }
                lock (_sum_SumSquare_Count_For_Each_Channel)
                {
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel] += sum;
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel + 1] += sumSquare;
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel + 2] += GetWidth()*GetHeight();
                }
            }
        }


        public static BitmapContent Resize(string path, int targetWidth, int targetHeight)
        {
            return CropAndResize(path, null, targetWidth, targetHeight);
        }

        public static BitmapContent CropAndResize(string path, Rectangle? croppedRectangle, int targetWidth, int targetHeight)
        {
            var settings = new ProcessImageSettings {Width = targetWidth, Height = targetHeight, SaveFormat =FileFormat.Bmp};
            if (croppedRectangle.HasValue)
            {
                settings.Crop = croppedRectangle.Value;
                settings.ResizeMode = CropScaleMode.Crop;
            }
            else
            {
                settings.ResizeMode = CropScaleMode.Stretch;
            }
            using var stream = new MemoryStream();
            MagicImageProcessor.ProcessImage(path, stream, settings);
            return ValueFromBmpContent(stream.ToArray(), targetWidth, targetHeight);
        }

        //public static BitmapContent CropAndResize(Bitmap bmp, Rectangle croppedRectangle, int targetWidth, int targetHeight, InterpolationMode interpolationMode = InterpolationMode.NearestNeighbor)
        //{
        //    if (interpolationMode != InterpolationMode.NearestNeighbor)
        //    {
        //        using var croppedBitmap = PictureTools.CropImage(bmp, croppedRectangle);
        //        using var resizedBitmap = PictureTools.ResizeImage(croppedBitmap, targetWidth, targetHeight, interpolationMode);
        //        return ValueFomSingleRgbBitmap(resizedBitmap);
        //    }

        //    var src = ValueFomSingleRgbBitmap(bmp);

        //    int destCountByChannel = targetHeight * targetWidth;
        //    var dest = new BitmapContent(new[] { src.Shape[0], targetHeight, targetWidth }, new byte[src.Shape[0] * destCountByChannel]);
        //    double zoomedWidth = ((double)targetWidth) / croppedRectangle.Width;
        //    double zoomedHeight = ((double)targetHeight) / croppedRectangle.Height;

        //    var srcPointer = src.SpanContent;
        //    var destPointer = dest.SpanContent;
        //    int destNexIdx = 0;
        //    for (int c = 0; c < dest.Shape[0]; ++c)
        //    {
        //        for (int rowDest = 0; rowDest < dest.Shape[1]; ++rowDest)
        //        {

        //            int rowSrc = croppedRectangle.Top + (int)(rowDest / zoomedHeight + 0.5);
        //            rowSrc = Math.Min(rowSrc, src.Shape[1] - 1);
        //            var srcShift = src.Idx(c, rowSrc, 0);

        //            for (int colDest = 0; colDest < dest.Shape[2]; ++colDest)
        //            {
        //                int colSrc = croppedRectangle.Left + (int)(colDest / zoomedWidth + 0.5);
        //                colSrc = Math.Min(colSrc, src.Shape[2] - 1);
        //                destPointer[destNexIdx++] = srcPointer[srcShift + colSrc];
        //            }
        //        }
        //    }
        //    //src.Dispose();
        //    return dest;
        //}


        private static BitmapContent ValueFromBmpContent(byte[] bmpData, int width, int height)
        {
            var shape = new[] { 3, height, width };
            var result = new BitmapContent(shape, null);
            var resultContent = result.SpanContent;
            int rowSizeInBytes = ((3*width + 3) / 4) * 4; //each row must have a size that is a multiple of 4
            var delta = bmpData.Length - rowSizeInBytes * height;
            int i = 0;
            for (int row = height-1; row >=0; --row)
            {
                for (int col = 0; col < width; col++)
                {
                    int j = delta+3 *col+row* rowSizeInBytes;
                    resultContent[i] = bmpData[j+2]; //R
                    resultContent[i + result.MultDim0] = bmpData[j + 1]; //G
                    resultContent[i + 2 * result.MultDim0] = bmpData[j+0]; //B
                    ++i;
                }
            }
            return result;
        }
        public static BitmapContent ValueFomSingleRgbBitmap(Bitmap bmp)
        {
            var width = bmp.Width;
            var height = bmp.Height;
            var rect = new Rectangle(0, 0, width, height);
            var bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var shape = new []{3, height, width};
            var result = new BitmapContent(shape, null);
            var stride = bmpData.Stride;
            var resultContent = result.SpanContent;
            unsafe
            {
                var imgPtr = (byte*)(bmpData.Scan0);
                int i = 0;
                for (int row = 0; row < height; row++)
                {
                    for (int col = 0; col < width; col++)
                    {
                        resultContent[i] = *(imgPtr + 2); //R
                        resultContent[i + result.MultDim0] = *(imgPtr + 1); //G
                        resultContent[i + 2 * result.MultDim0] = *(imgPtr + 0); //B
                        ++i;
                        imgPtr += 3;
                    }
                    imgPtr += stride - width * 3;
                }
            }
            // Unlock the bits.
            bmp.UnlockBits(bmpData);
            return result;
        }
        public bool IsDuplicate(BitmapContent b, double epsilon, RGBColorFactoryWithCache cache)
        {
            if (Count > b.Count)
            {
                return b.IsDuplicate(this, epsilon, cache);
            }

            var a = this;
            Debug.Assert(Count<=b.Count);
            int aIndex = 0;
            int aHeight = a.Shape[1];
            int aWidth = a.Shape[2];
            int bHeight = b.Shape[1];
            int bWidth = b.Shape[2];

            var aContent = a.SpanContent;
            var bContent = b.SpanContent;

            double distanceSum = 0;


            for (int aRow = 0; aRow < aHeight; aRow++)
            {
                int bRow = Math.Min((aRow * bHeight) / aHeight, bHeight - 1);
                for (int aCol = 0; aCol < aWidth; aCol++)
                {
                    var aColor = cache.Build(aContent[aIndex], aContent[aIndex + a.MultDim0], aContent[aIndex + 2 * a.MultDim0]);
                    int bCol = Math.Min((aCol * bWidth) / aWidth, bWidth - 1);
                    var bIndex = bCol + bRow * bWidth;
                    var bColor = cache.Build(bContent[bIndex], bContent[bIndex + b.MultDim0], bContent[bIndex + 2 * b.MultDim0]);
                    ++aIndex;
                    distanceSum += aColor.ColorDistance(bColor);
                    double currentAvgDistance = distanceSum / aIndex;
                    if (aIndex > 100 && currentAvgDistance > 5 * epsilon)
                    {
                        return false;
                    }
                }
            }
            double avgDistance = distanceSum / aIndex;
            return avgDistance < epsilon;
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
        /// <returns></returns>
        private static BitmapContent ValueFromSeveralSingleChannelBitmaps(IReadOnlyList<Bitmap> bmps)
        {
            var width = bmps[0].Width;
            var height = bmps[0].Height;
            var shape = new[] { bmps.Count, height, width };
            var result = new BitmapContent(shape, null);

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
                    var imgPtr = (byte*)(bmpData.Scan0);
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
    }
}
