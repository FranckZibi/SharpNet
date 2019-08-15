using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;

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
        /// <summary>
        /// % of the max(width,height) of the zero mask to apply to the input picture (see: https://arxiv.org/pdf/1708.04552.pdf)
        /// recommended size : 16/32=0.5 (= 16x16) for CIFAR10 / 8/32=0.25 (= 8x8) for CIFAR100 / 20/32 (= 20x20) for SVHN / 32/96 (= 32x32) for STL-10
        /// less or equal to 0.0 means no cutout
        /// </summary>
        private readonly double _cutoutPatchPercentage;
        /// <summary>
        /// % of the max(width,height) of the CutMix mask to apply to the input picture (see: https://arxiv.org/pdf/1905.04899.pdf)
        /// </summary>
        private readonly bool _CutMix;
        


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

        public static readonly ImageDataGenerator NoDataAugmentation = new ImageDataGenerator(0, 0, false, false, FillModeEnum.Nearest, 0.0, 0.0, false, 0.0, 0.0);

        public ImageDataGenerator(
            double widthShiftRangeInPercentage, double heightShiftRangeInPercentage, 
            bool horizontalFlip, bool verticalFlip, FillModeEnum fillMode, double fillModeConstantVal,
            double cutoutPatchPercentage,
            bool cutMix,
            double rotationRangeInDegrees,
            double zoomRange)
        {
            if (cutoutPatchPercentage > 0 && cutMix)
            {
                throw new ArgumentException("cutout and CutMix can't be used at the same time");
            }
            Debug.Assert(widthShiftRangeInPercentage >= 0);
            Debug.Assert(widthShiftRangeInPercentage <= 1.0);
            Debug.Assert(heightShiftRangeInPercentage >= 0);
            Debug.Assert(heightShiftRangeInPercentage <= 1.0);
            Debug.Assert(cutoutPatchPercentage <= 1.0);
            Debug.Assert(rotationRangeInDegrees >= 0);
            Debug.Assert(rotationRangeInDegrees <= 180.0);
            Debug.Assert(zoomRange >= 0);
            Debug.Assert(zoomRange <= 1.0);
            _widthShiftRangeInPercentage = widthShiftRangeInPercentage;
            _heightShiftRangeInPercentage = heightShiftRangeInPercentage;
            _horizontalFlip = horizontalFlip;
            _verticalFlip = verticalFlip;
            _fillMode = fillMode;
            _fillModeConstantVal = fillModeConstantVal;
            _cutoutPatchPercentage = cutoutPatchPercentage;
            _CutMix = cutMix;
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
        /// <param name="dataSetLoader"></param>
        /// <param name="elementId"></param>
        /// <param name="xInputBufferPictures"></param>
        /// <param name="elementIdToCategoryId"></param>
        /// <param name="yInputOneHot"></param>
        /// <param name="indexInBuffer"></param>
        /// <param name="xOutputBufferPictures"></param>
        /// <param name="yOutputOneHot"></param>
        /// <param name="typeSize"></param>
        public void CreateSingleInputForEpoch<T>(int epoch, bool isTraining, IDataSetLoader<T> dataSetLoader,
            int elementId, CpuTensor<T> xInputBufferPictures, int[] elementIdToCategoryId, CpuTensor<T> yInputOneHot, 
            int indexInBuffer, CpuTensor<T> xOutputBufferPictures, CpuTensor<T> yOutputOneHot, int typeSize) where T : struct
        {
            // NCHW tensors
            Debug.Assert(xInputBufferPictures.SameShape(xOutputBufferPictures));
            Debug.Assert(xInputBufferPictures.Shape.Length == xOutputBufferPictures.Shape.Length);
            //same number of channels
            Debug.Assert(xInputBufferPictures.Shape[1] == xOutputBufferPictures.Shape[1]);
            //same number of rows
            Debug.Assert(xInputBufferPictures.Shape[2] == xOutputBufferPictures.Shape[2]);
            //same number of columns
            Debug.Assert(xInputBufferPictures.Shape[3] == xOutputBufferPictures.Shape[3]);

            //We compute the output y vector
            var yInputIdx = yInputOneHot.Idx(elementId);
            var yOutputIdx = yOutputOneHot.Idx(indexInBuffer);
            Buffer.BlockCopy(yInputOneHot.Content, yInputIdx * typeSize, yOutputOneHot.Content, yOutputIdx * typeSize, yInputOneHot.MultDim0 * typeSize);
            /*
            var yOutputIdx = yOutputOneHot.Idx(indexInBuffer, elementIdToCategoryId[elementId]);
            if (yOutputOneHot.Content is double[])
            {
                (yOutputOneHot.Content as double[])[yOutputIdx] = 1.0;
            }
            else
            {
                // ReSharper disable once PossibleNullReferenceException
                (yOutputOneHot.Content as float[])[yOutputIdx] = 1.0f;
            }*/

            //We copy the element Id in the input buffer 'xInputBufferPictures' at index 'indexInBuffer'
            dataSetLoader.LoadAt(elementId, indexInBuffer, xInputBufferPictures);


            if (!UseDataAugmentation || (epoch == 1) || !isTraining)
            {
                //we'll just copy the input picture from index 'inputPictureIndex' in 'inputEnlargedPictures' to index 'outputPictureIndex' of 'outputBufferPictures'
                var pictureInputIdx = xInputBufferPictures.Idx(indexInBuffer);
                var pictureOutputIdx = xOutputBufferPictures.Idx(indexInBuffer);
                Buffer.BlockCopy(xInputBufferPictures.Content, pictureInputIdx * typeSize, xOutputBufferPictures.Content, pictureOutputIdx * typeSize, xOutputBufferPictures.MultDim0 * typeSize);
                return;
            }
            var rand = _rands[elementId % _rands.Length];

            //random shift
            int heightShiftRangeInPixels = GetPadding(xInputBufferPictures.Shape[2], _heightShiftRangeInPercentage);
            int widthShiftRangeInPixels = GetPadding(xInputBufferPictures.Shape[3], _widthShiftRangeInPercentage);
            var heightShift = rand.Next(2 * heightShiftRangeInPixels + 1) - heightShiftRangeInPixels;
            var widthShift = rand.Next(2 * widthShiftRangeInPixels + 1) - widthShiftRangeInPixels;

            //random horizontal and vertical flip
            var horizontalFlip = _horizontalFlip && rand.Next(2) == 0;
            var verticalFlip = _verticalFlip && rand.Next(2) == 0;

            //random zoom multiplier in range [1.0-zoomRange, 1.0+zoomRange]
            var zoom = 2*_zoomRange * rand.NextDouble() - _zoomRange;
            var widthMultiplier = (1.0+zoom);
            var heightMultiplier = (1.0+zoom);

            var nbRows = xOutputBufferPictures.Shape[2];
            var nbCols = xOutputBufferPictures.Shape[3];
            Cutout(nbRows, nbCols, rand, out var cutoutRowStart, out var cutoutRowEnd, out var cutoutColStart, out var cutoutColEnd);
            CutMix(nbRows, nbCols, rand, out var cutMixRowStart, out var cutMixRowEnd, out var cutMixColStart, out var cutMixColEnd, out double cutMixLambda);
            int inputPictureIndexForCutMix = rand.Next(xInputBufferPictures.Shape[0]);
            //random rotation in range [-_rotationRangeInDegrees, +_rotationRangeInDegrees]
            var rotationInDegrees = 2 * _rotationRangeInDegrees * rand.NextDouble() - _rotationRangeInDegrees;

            InitializeOutputPicture(
                xInputBufferPictures, 
                xOutputBufferPictures, indexInBuffer,
                widthShift, heightShift, _fillMode,
                horizontalFlip, verticalFlip,
                widthMultiplier, heightMultiplier,
                cutoutRowStart, cutoutRowEnd, cutoutColStart, cutoutColEnd,
                cutMixRowStart, cutMixRowEnd, cutMixColStart, cutMixColEnd, inputPictureIndexForCutMix,
                rotationInDegrees);


            if (_CutMix && elementIdToCategoryId[elementId]!= elementIdToCategoryId[inputPictureIndexForCutMix])
            {
                // We need to update the expected y using CutMix lambda
                // the associated y is:
                //      '1-cutMixLambda' % of the category of the element at 'inputPictureIndex'
                //        'cutMixLambda' % of the category of the element at 'inputPictureIndexForCutMix'
                if (yOutputOneHot.UseDoublePrecision)
                {
                    (yOutputOneHot as CpuTensor<double>)?.Set(indexInBuffer, elementIdToCategoryId[elementId], cutMixLambda);
                    (yOutputOneHot as CpuTensor<double>)?.Set(indexInBuffer, elementIdToCategoryId[inputPictureIndexForCutMix], 1.0 - cutMixLambda);
                }
                else
                {
                    (yOutputOneHot as CpuTensor<float>)?.Set(indexInBuffer, elementIdToCategoryId[elementId], (float)cutMixLambda);
                    (yOutputOneHot as CpuTensor<float>)?.Set(indexInBuffer, elementIdToCategoryId[inputPictureIndexForCutMix], (float)(1.0 - cutMixLambda));
                }
            }
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
            equals &= Utils.Equals(_CutMix, other._CutMix, id + ":_CutMix", ref errors);
            equals &= Utils.Equals(_rotationRangeInDegrees, other._rotationRangeInDegrees, epsilon, id + ":_rotationRangeInDegrees", ref errors);
            equals &= Utils.Equals(_zoomRange, other._zoomRange, epsilon, id + ":_zoomRange", ref errors);
            return equals;
        }
        public static void InitializeOutputPicture<T>(CpuTensor<T> xInputBufferPictures,
            CpuTensor<T> xOutputBufferPictures, int indexInBuffer,
            int widthShift, int heightShift, FillModeEnum _fillMode,
            bool horizontalFlip, bool verticalFlip,
            double widthMultiplier, double heightMultiplier,
            int cutoutRowStart, int cutoutRowEnd, int cutoutColStart, int cutoutColEnd,
            int cutMixRowStart, int cutMixRowEnd, int cutMixColStart, int cutMixColEnd, int inputPictureIndexForCutMix,
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
                    var outputPictureIdx = xOutputBufferPictures.Idx(indexInBuffer, channel, rowOutput, 0);
                    for (int colOutput = 0; colOutput < nbCols; ++colOutput)
                    {
                        //we check if we should apply cutout to the pixel
                        if (rowOutput >= cutoutRowStart && rowOutput <= cutoutRowEnd && colOutput >= cutoutColStart && colOutput <= cutoutColEnd)
                        {
                            xOutputBufferPictures[outputPictureIdx + colOutput] = default;
                            continue;
                        }

                        //we check if we should apply cutMix to the pixel
                        if (rowOutput >= cutMixRowStart && rowOutput <= cutMixRowEnd && colOutput >= cutMixColStart && colOutput <= cutMixColEnd)
                        {
                            xOutputBufferPictures[outputPictureIdx + colOutput] = xInputBufferPictures.Get(inputPictureIndexForCutMix, channel, rowOutput, colOutput);
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
                        rowInput = Math.Min(Math.Max(0, rowInput), nbRows - 1);
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
                        colInput = Math.Min(Math.Max(0, colInput), nbCols - 1);
                        Debug.Assert(colInput >= 0 && colInput < nbCols);
                        xOutputBufferPictures[outputPictureIdx + colOutput] = xInputBufferPictures.Get(indexInBuffer, channel, rowInput, colInput);
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
                .Add(nameof(_CutMix), _CutMix)
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
                (bool)serialized[nameof(_CutMix)],
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
                throw new ArgumentException("invalid _cutoutPatchPercentage:" + _cutoutPatchPercentage);
            }
            int cutoutPatchLength = (int)Math.Round(_cutoutPatchPercentage * Math.Max(nbRows, nbCols), 0.0);
            //the cutout patch will be centered at (rowMiddle,colMiddle)
            //its size will be between '1x1' (minimum patch size if the center is a corner) to 'cutoutPatchLength x cutoutPatchLength' (maximum size)
            var rowMiddle = rand.Next(nbRows);
            var colMiddle = rand.Next(nbCols);

            //Tested on 10-aug-2019: -10bps
            //rowStart = Math.Max(0, rowMiddle - cutoutPatchLength / 2);
            //rowEnd = Math.Min(nbRows - 1, rowMiddle + cutoutPatchLength/2 - 1);
            //colStart = Math.Max(0, colMiddle - cutoutPatchLength / 2);
            //colEnd = Math.Min(nbCols - 1, colMiddle + cutoutPatchLength/2 - 1);

            //TODO ADD TEST: Cutout of always the max dimension

            rowStart = Math.Max(0, rowMiddle - cutoutPatchLength / 2);
            rowEnd = Math.Min(nbRows - 1, rowStart + cutoutPatchLength - 1);
            colStart = Math.Max(0, colMiddle - cutoutPatchLength / 2);
            colEnd = Math.Min(nbCols - 1, colStart + cutoutPatchLength - 1);
        }
        private void CutMix(int nbRows, int nbCols, Random rand, out int rowStart, out int rowEnd, out int colStart, out int colEnd, out double lambda)
        {
            if (!_CutMix)
            {
                rowStart = rowEnd = colStart = colEnd = -1;
                lambda = 0.0;
                return;
            }

            lambda = rand.NextDouble();
            var cutMixHeight = (int)(nbRows * Math.Sqrt(1.0 - lambda));
            var cutMixWidth = (int)(nbCols * Math.Sqrt(1.0 - lambda));

            //the cutout patch will be centered at (rowMiddle,colMiddle)
            //its size will be between '1x1' (minimum patch size if the center is a corner) to 'cutoutPatchLength x cutoutPatchLength' (maximum size)
            var rowMiddle = rand.Next(nbRows);
            var colMiddle = rand.Next(nbCols);
            rowStart = Math.Max(0, rowMiddle - cutMixHeight / 2);
            rowEnd = Math.Min(nbRows - 1, rowMiddle + cutMixHeight/2 - 1);
            colStart = Math.Max(0, colMiddle - cutMixWidth/2 / 2);
            colEnd = Math.Min(nbCols - 1, colMiddle + cutMixWidth/2 - 1);
            lambda = 1.0-((double)((rowEnd - rowStart + 1) * (colEnd - colStart + 1))) / (nbCols * nbRows);
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
