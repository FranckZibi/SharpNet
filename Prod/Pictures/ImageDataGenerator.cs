using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.Data;

namespace SharpNet.Pictures
{
    public class ImageDataGenerator
    {
        //TODO: add FillModeEnum: Constant
        public enum FillModeEnum { Nearest, Reflect };

        #region private fields
        private readonly Random[] _rands;
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
        //see https://arxiv.org/pdf/1708.04552.pdf
        /// <summary>
        /// % of the max(width,height) of the zero mask to apply to the input picture (see  https://arxiv.org/pdf/1708.04552.pdf)
        /// recommended size : 16/32=0.5 (= 16x16) for CIFAR10 / 8/32=0.25 (= 8x8) for CIFAR100 / 20/32 (= 20x20) for SVHN / 32/96 (= 32x32) for STL-10
        /// less or equal to 0.0 means no cutout
        /// </summary>
        private readonly double _cutoutPatchPercentage;
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

        public static readonly ImageDataGenerator NoDataAugmentation = new ImageDataGenerator(0, 0, false, false, FillModeEnum.Nearest, 0.0, 0.0, 0.0, 0.0);

        public ImageDataGenerator(
            double widthShiftRangeInPercentage, double heightShiftRangeInPercentage, 
            bool horizontalFlip, bool verticalFlip, FillModeEnum fillMode, double fillModeConstantVal,
            double cutoutPatchPercentage,
            double rotationRangeInDegrees,
            double zoomRange)
        {
            Debug.Assert(widthShiftRangeInPercentage >= 0);
            Debug.Assert(widthShiftRangeInPercentage <= 1.0);
            Debug.Assert(heightShiftRangeInPercentage >= 0);
            Debug.Assert(heightShiftRangeInPercentage <= 1.0);
            Debug.Assert(zoomRange >= 0);
            Debug.Assert(zoomRange <= 1.0);
            Debug.Assert(rotationRangeInDegrees >= 0);
            Debug.Assert(rotationRangeInDegrees <= 180.0);
            _widthShiftRangeInPercentage = widthShiftRangeInPercentage;
            _heightShiftRangeInPercentage = heightShiftRangeInPercentage;
            _horizontalFlip = horizontalFlip;
            _verticalFlip = verticalFlip;
            _fillMode = fillMode;
            _fillModeConstantVal = fillModeConstantVal;
            _cutoutPatchPercentage = cutoutPatchPercentage;
            _rotationRangeInDegrees = rotationRangeInDegrees;
            _zoomRange = zoomRange;
            _rands = new Random[2 * Environment.ProcessorCount];
            for (int i = 0; i < _rands.Length; ++i)
            {
                _rands[i] = new Random(i);
            }
        }


        /// <summary>
        /// takes the input (enlarged) picture at index 'inputPictureIndex' in 'inputEnlargedPictures'&'yInputOneHot'
        /// and stores a data augmented version of it at index 'outputPictureIndex' of 'outputBufferPictures'&'yOutputOneHot'
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="epoch"></param>
        /// <param name="isTraining"></param>
        /// <param name="xInputPictures"></param>
        /// <param name="yInputOneHot"></param>
        /// <param name="xOutputBufferPictures"></param>
        /// <param name="yOutputOneHot"></param>
        /// <param name="inputPictureIndex"></param>
        /// <param name="outputPictureIndex"></param>
        /// <param name="typeSize"></param>
        public void CreateSingleInputForEpoch<T>(int epoch, bool isTraining, CpuTensor<T> xInputPictures, CpuTensor<T> yInputOneHot, CpuTensor<T> xOutputBufferPictures, CpuTensor<T> yOutputOneHot, int inputPictureIndex, int outputPictureIndex, int typeSize) where T : struct
        {
            // NCHW tensors
            Debug.Assert(xInputPictures.Shape.Length == 4);
            Debug.Assert(xInputPictures.Shape.Length == xOutputBufferPictures.Shape.Length);
            //same number of channels
            Debug.Assert(xInputPictures.Shape[1] == xOutputBufferPictures.Shape[1]);
            //same number of rows
            Debug.Assert(xInputPictures.Shape[2] == xOutputBufferPictures.Shape[2]);
            //same number of columns
            Debug.Assert(xInputPictures.Shape[3] == xOutputBufferPictures.Shape[3]);

            //We compute the output y vector
            var yInputIdx = yInputOneHot.Idx(inputPictureIndex);
            var yOutputIdx = yOutputOneHot.Idx(outputPictureIndex);
            Buffer.BlockCopy(yInputOneHot.Content, yInputIdx * typeSize, yOutputOneHot.Content, yOutputIdx * typeSize, yInputOneHot.MultDim0 * typeSize);

            if (!UseDataAugmentation || (epoch == 1) || !isTraining)
            {
                //we'll just copy the input picture from index 'inputPictureIndex' in 'inputEnlargedPictures' to index 'outputPictureIndex' of 'outputBufferPictures'
                var pictureInputIdx = xInputPictures.Idx(inputPictureIndex);
                var pictureOutputIdx = xOutputBufferPictures.Idx(outputPictureIndex);
                Buffer.BlockCopy(xInputPictures.Content, pictureInputIdx * typeSize, xOutputBufferPictures.Content, pictureOutputIdx * typeSize, xOutputBufferPictures.MultDim0 * typeSize);
                return;
            }
            var rand = _rands[inputPictureIndex % _rands.Length];
            var horizontalFlip = _horizontalFlip && rand.Next(2) == 0;
            var verticalFlip = _verticalFlip && rand.Next(2) == 0;
            var nbRows = xOutputBufferPictures.Shape[2];
            var nbCols = xOutputBufferPictures.Shape[3];
            Cutout(nbRows, nbCols, rand, out var cutoutRowStart, out var cutoutRowEnd, out var cutoutColStart, out var cutoutColEnd);
            int heightShiftRangeInPixels = GetPadding(xInputPictures.Shape[2], _heightShiftRangeInPercentage);
            int widthShiftRangeInPixels = GetPadding(xInputPictures.Shape[3], _widthShiftRangeInPercentage);
            var heightShift = rand.Next(2 * heightShiftRangeInPixels + 1) - heightShiftRangeInPixels;
            var widthShift = rand.Next(2 * widthShiftRangeInPixels + 1) - widthShiftRangeInPixels;
            var rotationInDegrees = 2*_rotationRangeInDegrees*rand.NextDouble() - _rotationRangeInDegrees;
            var zoom = 2*_zoomRange * rand.NextDouble() - _zoomRange;
            var widthMultiplier = (1.0+zoom);
            var heightMultiplier = (1.0+zoom);

            InitializeOutputPicture(
                xInputPictures, inputPictureIndex, 
                xOutputBufferPictures, outputPictureIndex,
                widthShift, heightShift, _fillMode,
                horizontalFlip, verticalFlip,
                widthMultiplier, heightMultiplier,
                cutoutRowStart, cutoutRowEnd, cutoutColStart, cutoutColEnd, 
                rotationInDegrees);
        }
        public bool Equals(ImageDataGenerator other, double epsilon, string id, ref string errors)
        {
            var equals = true;
            equals &= Utils.Equals(_widthShiftRangeInPercentage, other._widthShiftRangeInPercentage, epsilon, id + ":_widthShiftRange", ref errors);
            equals &= Utils.Equals(_heightShiftRangeInPercentage, other._heightShiftRangeInPercentage, epsilon, id + ":_heightShiftRange", ref errors);
            equals &= Utils.Equals(_horizontalFlip, other._horizontalFlip, id + ":_horizontalFlip", ref errors);
            equals &= Utils.Equals(_verticalFlip, other._verticalFlip, id + ":_verticalFlip", ref errors);
            equals &= Utils.Equals(_fillMode, other._fillMode, id + ":_fillMode", ref errors);
            equals &= Utils.Equals(_fillModeConstantVal, other._fillModeConstantVal, epsilon, id + ":_fillModeConstantVal", ref errors);
            equals &= Utils.Equals(_cutoutPatchPercentage, other._cutoutPatchPercentage, epsilon, id + ":_cutoutPatchPercentage", ref errors);
            equals &= Utils.Equals(_rotationRangeInDegrees, other._rotationRangeInDegrees, epsilon, id + ":_rotationRangeInDegrees", ref errors);
            equals &= Utils.Equals(_zoomRange, other._zoomRange, epsilon, id + ":_zoomRange", ref errors);
            return equals;
        }
        public static void InitializeOutputPicture<T>(CpuTensor<T> xInputPictures, int inputPictureIndex,
         CpuTensor<T> xOutputBufferPictures, int outputPictureIndex,
         int widthShift, int heightShift, FillModeEnum _fillMode,
         bool horizontalFlip, bool verticalFlip,
         double widthMultiplier, double heightMultiplier,
         int cutoutRowStart, int cutoutRowEnd, int cutoutColStart, int cutoutColEnd, 
         double rotationInDegrees) where T : struct
        {
            var nbRows = xOutputBufferPictures.Shape[2];
            var nbCols = xOutputBufferPictures.Shape[3];
            var transformer = new PointTransformer(heightShift, widthShift, horizontalFlip, verticalFlip,
                widthMultiplier, heightMultiplier, rotationInDegrees, nbRows, nbCols);

            for (int channel = 0; channel < xOutputBufferPictures.Shape[1]; ++channel)
            {
                for (int rowOutput = 0; rowOutput < nbRows; ++rowOutput)
                {
                    var outputPictureIdx = xOutputBufferPictures.Idx(outputPictureIndex, channel, rowOutput, 0);
                    for (int colOutput = 0; colOutput < nbCols; ++colOutput)
                    {
                        //we check if we should apply cutout to the pixel
                        if (rowOutput >= cutoutRowStart && rowOutput <= cutoutRowEnd &&
                            colOutput >= cutoutColStart && colOutput <= cutoutColEnd)
                        {
                            xOutputBufferPictures[outputPictureIdx + colOutput] = default;
                            continue;
                        }
                        
                        var rowInput = transformer.UnconvertRow(rowOutput, colOutput);
                        if (rowInput < 0)
                        {
                            rowInput = _fillMode == FillModeEnum.Reflect ? Math.Abs(rowInput + 1) : 0;
                        }
                        if (rowInput >= nbRows)
                        {
                            rowInput = _fillMode == FillModeEnum.Reflect ? (nbRows - 1 - (rowInput - nbRows)) : (nbRows - 1);
                        }
                        rowInput = Math.Max(0, rowInput);
                        Debug.Assert(rowInput >= 0 && rowInput < nbRows);

                        var colInput = transformer.UnconvertCol(rowOutput, colOutput);
                        if (colInput < 0)
                        {
                            colInput = _fillMode == FillModeEnum.Reflect ? Math.Abs(colInput + 1) : 0;
                        }
                        if (colInput >= nbCols)
                        {
                            colInput = _fillMode == FillModeEnum.Reflect ? (nbCols - 1 - (colInput - nbCols)) : (nbCols - 1);
                        }
                        colInput = Math.Max(0, colInput);
                        Debug.Assert(colInput >= 0 && colInput < nbCols);
                        xOutputBufferPictures[outputPictureIdx + colOutput] = xInputPictures.Get(inputPictureIndex, channel, rowInput, colInput);
                    }
                }
            }
        }


        #region serialization
        public string Serialize()
        {
            if (!UseDataAugmentation)
            {
                return "";
            }
            return new Serializer()
                .Add(nameof(_widthShiftRangeInPercentage), _widthShiftRangeInPercentage)
                .Add(nameof(_heightShiftRangeInPercentage), _heightShiftRangeInPercentage)
                .Add(nameof(_horizontalFlip), _horizontalFlip)
                .Add(nameof(_verticalFlip), _verticalFlip)
                .Add(nameof(_fillMode), (int)_fillMode)
                .Add(nameof(_fillModeConstantVal), _fillModeConstantVal)
                .Add(nameof(_cutoutPatchPercentage), _cutoutPatchPercentage)
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
                (double)serialized[nameof(_widthShiftRangeInPercentage)],
                (double)serialized[nameof(_heightShiftRangeInPercentage)],
                (bool)serialized[nameof(_horizontalFlip)],
                (bool)serialized[nameof(_verticalFlip)],
                (FillModeEnum)serialized[nameof(_fillMode)],
                (double)serialized[nameof(_fillModeConstantVal)],
                (double)serialized[nameof(_cutoutPatchPercentage)],
                (double)serialized[nameof(_rotationRangeInDegrees)],
                (double)serialized[nameof(_zoomRange)]
                );
        }
        #endregion
 
        private void Cutout(int nbRows, int nbCols, Random rand, out int rowStart, out int rowEnd, out int colStart, out int colEnd)
        {
            if (_cutoutPatchPercentage <= 0)
            {
                rowStart = rowEnd = colStart = colEnd = -1;
                return;
            }
            if (_cutoutPatchPercentage > 1.0)
            {
                throw new Exception("invalid _cutoutPatchPercentage:" + _cutoutPatchPercentage);
            }
            int cutoutPatchLength = (int)Math.Round(_cutoutPatchPercentage * Math.Max(nbRows, nbCols), 0.0);
            //the cutout patch will be centered at (rowMiddle,colMiddle)
            //its size will be between '1x1' (minimum patch size if the center is a corner) to 'cutoutPatchLength x cutoutPatchLength' (maximum size)
            var rowMiddle = rand.Next(nbRows);
            var colMiddle = rand.Next(nbCols);
            rowStart = Math.Max(0, rowMiddle - cutoutPatchLength / 2);
            rowEnd = Math.Min(nbRows - 1, rowStart + cutoutPatchLength - 1);
            colStart = Math.Max(0, colMiddle - cutoutPatchLength / 2);
            colEnd = Math.Min(nbCols - 1, colStart + cutoutPatchLength - 1);
        }
        private bool UseDataAugmentation => !ReferenceEquals(this, NoDataAugmentation);
        private static int GetPadding(int pictureWidth, double widthShiftRange)
        {
            if (widthShiftRange <= 0)
            {
                return 0;
            }
            return (int)Math.Ceiling(pictureWidth * widthShiftRange);
        }
    }
}
