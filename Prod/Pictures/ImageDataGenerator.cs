using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using SharpNet.CPU;
using SharpNet.Data;

namespace SharpNet.Pictures
{
    public class ImageDataGenerator
    {
        public enum FillModeEnum {Nearest,Reflect};

        #region private fields
        private readonly Random _randForShuffle;
        private readonly Random[] _rands;
        //randomly shift images horizontally
        private readonly double _widthShiftRange;
        //randomly shift images vertically
        private readonly double _heightShiftRange;
        //set mode for filling points outside the input boundaries
        private readonly FillModeEnum _fillMode;
        //value used for fill_mode  {get;set;} = "constant"
        private readonly double _fillModeConsantVal;
        //randomly flip images
        private readonly bool _horizontalFlip;
        //randomly flip images
        private readonly bool _verticalFlip;
        //set input mean to 0 over the dataset
        //public bool FeaturewiseCenter { get; set; } = false;
        //set each sample mean to 0
        //public bool SamplewiseCenter { get; set; } = false;
        //divide inputs by std of dataset
        //public bool FeaturewiseStdNormalization { get; set; } = false;
        //divide each input by its std
        //public bool SamplewiseStdNormalization { get; set; } = false;
        //apply ZCA whitening
        //public bool ZcaWhitening { get; set; } = false;
        //epsilon for ZCA whitening
        //public double ZcaEpsilon { get; set; } = 1e-06;
        //randomly rotate images in the range (deg 0 to 180)
        //public double RotationRange { get; set; } = 0;
        //set range for random shear
        //public double ShearRange { get; set; } = 0;
        //set range for random zoom
        //public double ZoomRange { get; set; } = 0;
        //set range for random channel shifts
        //public double ChannelShiftRange { get; set; } = 0;
        //set rescaling factor (applied before any other transformation)
        //public string Rescale { get; set; } = "None";
        //set function that will be applied on each input
        //public string PreprocessingFunction { get; set; } = "None";
        //image data format; either "channels_first" or "channels_last"
        //public string DataFormat { get; set; } = "None";
        //fraction of images reserved for validation (strictly between 0 and 1)
        //public double ValidationSplit { get; set; } = 0.0;
        #endregion
        public static readonly ImageDataGenerator NoDataAugmentation = new ImageDataGenerator(0, 0, false, false, FillModeEnum.Nearest, 0.0);

        public ImageDataGenerator(double widthShiftRange, double heightShiftRange, bool horizontalFlip, bool verticalFlip, FillModeEnum fillMode, double fillModeConsantVal)
        {
            _widthShiftRange = widthShiftRange;
            _heightShiftRange = heightShiftRange;
            _horizontalFlip = horizontalFlip;
            _verticalFlip = verticalFlip;
            _fillMode = fillMode;
            _fillModeConsantVal = fillModeConsantVal;
            _randForShuffle = new Random(0);
            _rands = new Random[2 * Environment.ProcessorCount];
            for (int i = 0; i < _rands.Length; ++i)
            {
                _rands[i] = new Random(i);
            }
        }
        public void CreateInputForEpoch<T>(CpuTensor<T> inputEnlargedPictures, CpuTensor<T> yInputOneHot,  CpuTensor<T> outputBufferPictures, CpuTensor<T> yOutputOneHot, bool randomizeOrder) where T : struct
        {
            Debug.Assert(yInputOneHot.SameShape(yOutputOneHot));
            Debug.Assert(yInputOneHot.Dimension == 2);
            //same batch size
            Debug.Assert(inputEnlargedPictures.Shape[0] == outputBufferPictures.Shape[0]);
            Debug.Assert(inputEnlargedPictures.Shape[0] == yInputOneHot.Shape[0]);
            //same number of channels
            Debug.Assert(inputEnlargedPictures.Shape[1] == outputBufferPictures.Shape[1]);
            CheckPictures(outputBufferPictures, inputEnlargedPictures);

            var entireBatchSize = inputEnlargedPictures.Shape[0];
            var shuffledRows = Enumerable.Range(0, entireBatchSize).ToList();
            if (randomizeOrder)
            {
                Utils.Shuffle(shuffledRows, _randForShuffle);
            }
            var typeSize = Marshal.SizeOf(typeof(T));
            Parallel.For(0, entireBatchSize, inputPictureIndex => CreateSingleInputForEpoch(inputEnlargedPictures, yInputOneHot, outputBufferPictures, yOutputOneHot, inputPictureIndex, shuffledRows[inputPictureIndex], typeSize));
        }
        public CpuTensor<T> EnlargePictures<T>(CpuTensor<T> originalPicture) where T : struct
        {
            if (!UseDataAugmentation)
            {
                return originalPicture;
            }
            var n = originalPicture.Shape[0];
            var h = originalPicture.Shape[2];
            var w = originalPicture.Shape[3];
            GetPadding(h, w, out int paddingForTopAndBottom, out int paddingForLeftAndRight);
            var shapeOutput = (int[])originalPicture.Shape.Clone();
            var hOutput = h + 2 * paddingForTopAndBottom;
            var wOutput = w + 2 * paddingForLeftAndRight;
            shapeOutput[2] = hOutput;
            shapeOutput[3] = wOutput;
            var enlargedPictures = new CpuTensor<T>(shapeOutput, "enlargedPictures");
            int typeSize = Marshal.SizeOf(typeof(T));
            Parallel.For(0, n, pictureIndexToEnlarge => CreateSingleEnlargedPicture(originalPicture, enlargedPictures, typeSize, pictureIndexToEnlarge));
            return enlargedPictures;
        }

        #region serialization
        public string Serialize()
        {
            if (!UseDataAugmentation)
            {
                return "";
            }
            return new Serializer()
                .Add(nameof(_widthShiftRange), _widthShiftRange)
                .Add(nameof(_heightShiftRange), _heightShiftRange)
                .Add(nameof(_horizontalFlip), _horizontalFlip)
                .Add(nameof(_verticalFlip), _verticalFlip)
                .Add(nameof(_fillMode), (int)_fillMode)
                .Add(nameof(_fillModeConsantVal), _fillModeConsantVal)
                .ToString();
        }
        public static ImageDataGenerator ValueOf(IDictionary<string, object> serialized)
        {
            if (!serialized.ContainsKey(nameof(_widthShiftRange)))
            {
                return NoDataAugmentation;
            }
            return new ImageDataGenerator(
                (double) serialized[nameof(_widthShiftRange)],
                (double) serialized[nameof(_heightShiftRange)],
                (bool) serialized[nameof(_horizontalFlip)],
                (bool) serialized[nameof(_verticalFlip)],
                (FillModeEnum) serialized[nameof(_fillMode)],
                (double) serialized[nameof(_fillModeConsantVal)]
                );
        }
        #endregion

        private void CreateSingleInputForEpoch<T>(CpuTensor<T> inputEnlargedPictures, CpuTensor<T> yInputOneHot, CpuTensor<T> outputBufferPictures, CpuTensor<T> yOutputOneHot, int inputPictureIndex, int outputPictureIndex, int typeSize) where T : struct
        {
            var yInputIdx = yInputOneHot.Idx(inputPictureIndex);
            var yOutputIdx = yOutputOneHot.Idx(outputPictureIndex);
            Buffer.BlockCopy(yInputOneHot.Content, yInputIdx * typeSize, yOutputOneHot.Content, yOutputIdx * typeSize, yInputOneHot.MultDim0 * typeSize);

            if (!UseDataAugmentation)
            {
                //we'll just copy the input picture from index 'inputPictureIndex' in 'inputEnlargedPictures' to index 'outputPictureIndex' of 'outputBufferPictures'
                var pictureInputIdx = inputEnlargedPictures.Idx(inputPictureIndex);
                var pictureOutputIdx = outputBufferPictures.Idx(outputPictureIndex);
                Buffer.BlockCopy(inputEnlargedPictures.Content, pictureInputIdx * typeSize, outputBufferPictures.Content, pictureOutputIdx * typeSize, outputBufferPictures.MultDim0 * typeSize);
                return;
            }

            var hOutput = outputBufferPictures.Shape[2];
            var wOutput = outputBufferPictures.Shape[3];
            var wSizeInBytes = wOutput * typeSize;
            var paddingForTopAndBottom = (inputEnlargedPictures.Shape[2] - hOutput) / 2;
            var paddingForLeftAndRight = (inputEnlargedPictures.Shape[3] - wOutput) / 2;
            var rand = _rands[inputPictureIndex % _rands.Length];
            var rowInEnlargedPictures = rand.Next(2 * paddingForTopAndBottom);
            var colInEnlargedPictures = rand.Next(2 * paddingForLeftAndRight);
            var horizontalFlip = _horizontalFlip && rand.Next(2) == 0;
            var verticalFlip = _verticalFlip && rand.Next(2) == 0;
            for (int channel = 0; channel < outputBufferPictures.Shape[1]; ++channel)
            {
                for (int rowOutput = 0; rowOutput < hOutput; ++rowOutput)
                {
                    var rowInput = verticalFlip ? (rowInEnlargedPictures + hOutput - 1 - rowOutput) : (rowOutput + rowInEnlargedPictures);
                    var inputPictureIdx = inputEnlargedPictures.Idx(inputPictureIndex, channel, rowInput, colInEnlargedPictures);
                    var outputPictureIdx = outputBufferPictures.Idx(outputPictureIndex, channel, rowOutput, 0);
                    Buffer.BlockCopy(inputEnlargedPictures.Content, inputPictureIdx * typeSize, outputBufferPictures.Content, outputPictureIdx * typeSize, wSizeInBytes);
                    if (horizontalFlip)
                    {
                        Array.Reverse(outputBufferPictures.Content, outputPictureIdx, wOutput);
                    }
                }
            }
        }
        private bool UseDataAugmentation => !ReferenceEquals(this, NoDataAugmentation);
        private void GetPadding(int pictureHeight, int pictureWidth, out int paddingForTopAndBottom, out int paddingForLeftAndRight)
        {
            paddingForTopAndBottom = GetPadding(pictureHeight, _heightShiftRange);
            paddingForLeftAndRight = GetPadding(pictureWidth, _widthShiftRange);
        }
        private static int GetPadding(int pictureWidth, double widthShiftRange)
        {
            if (widthShiftRange <= 0)
            {
                return 0;
            }
            return (int)Math.Ceiling(pictureWidth * widthShiftRange);
        }
        private void CreateSingleEnlargedPicture<T>(CpuTensor<T> originalPictures, CpuTensor<T> enlargedPictures, int typeSize, int pictureIndex) where T : struct
        {
            CheckPictures(originalPictures, enlargedPictures);
            var h = originalPictures.Shape[2];
            var w = originalPictures.Shape[3];
            var wSizeInBytes = w*typeSize;
            var hEnlarged = enlargedPictures.Shape[2];
            var wEnlarged = enlargedPictures.Shape[3];
            var wEnlargedSizeInBytes = wEnlarged*typeSize;
            var paddingForTopAndBottom = (hEnlarged - h) / 2;
            var paddingForLeftAndRight = (wEnlarged - w) / 2;
            var enlargedPicturesContent = enlargedPictures.Content;
            for (int channel = 0; channel < originalPictures.Shape[1]; ++channel)
            {
                for (int row = 0; row < h; ++row)
                {
                    int originalIdx = originalPictures.Idx(pictureIndex, channel, row, 0);
                    int outputIdx = enlargedPictures.Idx(pictureIndex, channel, row + paddingForTopAndBottom, paddingForLeftAndRight);
                    Buffer.BlockCopy(originalPictures.Content, originalIdx * typeSize, enlargedPicturesContent, outputIdx * typeSize, wSizeInBytes);

                    //left of line
                    var startOfLineValue = enlargedPicturesContent[outputIdx];
                    for (int col = 1; col <= paddingForLeftAndRight; ++col)
                    {
                        enlargedPicturesContent[outputIdx - col] = (_fillMode==FillModeEnum.Reflect)?enlargedPicturesContent[outputIdx + col]: startOfLineValue;
                    }

                    //right of line
                    var endOfLineValue = enlargedPicturesContent[outputIdx+h-1];
                    for (int col = 1; col <= paddingForLeftAndRight; ++col)
                    {
                        enlargedPicturesContent[outputIdx + h - 1 + col] = (_fillMode == FillModeEnum.Reflect)?enlargedPicturesContent[outputIdx + h - 1-col]: endOfLineValue;
                    }
                }

                //top 
                int srcTopRowIdx = enlargedPictures.Idx(pictureIndex, channel, paddingForTopAndBottom, 0);
                for (int row = 1; row <= paddingForTopAndBottom; ++row)
                {
                    int srcIdx = (_fillMode == FillModeEnum.Reflect) ? enlargedPictures.Idx(pictureIndex, channel, paddingForTopAndBottom+row, 0): srcTopRowIdx;
                    int targetIdx = enlargedPictures.Idx(pictureIndex, channel, paddingForTopAndBottom - row, 0);
                    Buffer.BlockCopy(enlargedPictures.Content, srcIdx*typeSize, enlargedPictures.Content, targetIdx * typeSize, wEnlargedSizeInBytes);
                }

                //bottom 
                int srcBottomRowIdx = enlargedPictures.Idx(pictureIndex, channel, paddingForTopAndBottom + h - 1, 0);
                for (int row = 1; row <= paddingForTopAndBottom; ++row)
                {
                    int srcIdx = (_fillMode == FillModeEnum.Reflect) ? enlargedPictures.Idx(pictureIndex, channel, paddingForTopAndBottom + h - 1-row, 0) : srcBottomRowIdx;
                    int targetIdx = enlargedPictures.Idx(pictureIndex, channel, paddingForTopAndBottom + h - 1 + row, 0);
                    Buffer.BlockCopy(enlargedPicturesContent, srcIdx * typeSize, enlargedPicturesContent, targetIdx * typeSize, wEnlargedSizeInBytes);
                }
            }
        }
        private static void CheckPictures<T>(CpuTensor<T> originalPictures, CpuTensor<T> enlargedPictures) where T : struct
        {
            Debug.Assert(originalPictures.Shape.Length == 4);
            Debug.Assert(originalPictures.Shape.Length == enlargedPictures.Shape.Length);
            Debug.Assert(enlargedPictures.Shape[0] == originalPictures.Shape[0]);
            Debug.Assert(enlargedPictures.Shape[1] == originalPictures.Shape[1]);
            Debug.Assert(enlargedPictures.Shape[2] >= originalPictures.Shape[2]);
            Debug.Assert((enlargedPictures.Shape[2] - originalPictures.Shape[2]) % 2 == 0);
            Debug.Assert(enlargedPictures.Shape[3] >= originalPictures.Shape[3]);
            Debug.Assert((enlargedPictures.Shape[3] - originalPictures.Shape[3]) % 2 == 0);
            Debug.Assert((enlargedPictures.Shape[2] - originalPictures.Shape[2]) / 2 <= enlargedPictures.Shape[2]);
            Debug.Assert((enlargedPictures.Shape[3] - originalPictures.Shape[3]) / 2 <= enlargedPictures.Shape[3]);
        }
    }
}
